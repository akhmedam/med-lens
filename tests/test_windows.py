"""Token coverage, section isolation, source spans, and model input limits."""

import io
import json
import re
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from medlens.retrieval.windows import WindowConfig, WindowError, article_windows, build_windows, load_windows
from test_evaluation import synthetic_inputs


class Tokenizer:
    """Whitespace-word test double; real WordPiece is checked in the pilot run."""
    cls_token_id, sep_token_id = 101, 102

    def __init__(self):
        self.vocab = {}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r"\S+", text))
        for match in matches:
            self.vocab.setdefault(match.group(), len(self.vocab) + 1000)
        result = {"input_ids": [self.vocab[m.group()] for m in matches]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return result


def article(title="Title", sections=()):
    return {"pmid": "1", "title": title, "abstract": {"sections": [
        {"index": i, "label": f"Section {i}", "text": text, "source_path": f"/AbstractText[{i+1}]"}
        for i, text in enumerate(sections)]}, "text_version": {"text_version_id": "version-1"},
        "provenance": {"snapshot_id": "synthetic", "record_path": "/Article[1]"}}


class WindowTests(unittest.TestCase):
    def test_overlap_exact_boundaries_and_complete_section_token_coverage(self):
        for length, count in ((384, 1), (385, 2), (704, 2), (705, 3)):
            with self.subTest(length=length):
                source = article(sections=[" ".join(f"word{i}" for i in range(length))])
                rows = article_windows(source, Tokenizer())
                self.assertEqual(len(rows), count)
                covered = set()
                for index, row in enumerate(rows):
                    covered.update(range(row["token_start"], row["token_end"]))
                    self.assertLessEqual(row["body_token_count"], 384)
                    self.assertLessEqual(row["input_token_count"], 512)
                    if index:
                        self.assertEqual(rows[index-1]["token_end"]-row["token_start"],64)
                self.assertEqual(covered,set(range(length)))

    def test_sections_never_mix_and_source_offsets_preserve_original_unicode(self):
        source = article(sections=["Évidence\n  not improved", "A second section"])
        rows = article_windows(source, Tokenizer())
        self.assertEqual([r["section_index"] for r in rows],[0,1])
        for row in rows:
            original = source["abstract"]["sections"][row["section_index"]]["text"]
            self.assertEqual(row["text"], original[row["char_start"]:row["char_end"]])
            self.assertEqual(row["provenance"],source["provenance"])
        self.assertIn("Évidence\n  not",rows[0]["text"])

    def test_title_budget_reduces_body_capacity_without_dropping_tokens(self):
        source = article(title=" ".join(["title"]*50),sections=[" ".join(["body"]*100)])
        rows = article_windows(source, Tokenizer(), WindowConfig(80,10,100))
        self.assertEqual([r["body_token_count"] for r in rows],[48,48,24])
        self.assertEqual(rows[0]["input_token_count"],100)
        self.assertEqual(rows[0]["input_ids"][0],101)
        self.assertEqual(rows[0]["input_ids"][-1],102)

    def test_long_title_errors_only_when_required_overlap_cannot_fit(self):
        title=" ".join(["title"]*70)
        self.assertEqual(len(article_windows(article(title,["short"]),Tokenizer(),WindowConfig(80,64,100))),1)
        with self.assertRaisesRegex(WindowError,"overlap"):
            article_windows(article(title,[" ".join(["word"]*50)]),Tokenizer(),WindowConfig(80,64,100))
        with self.assertRaisesRegex(WindowError,"title alone"):
            article_windows(article(" ".join(["title"]*100)),Tokenizer(),WindowConfig(80,10,100))

    def test_title_only_missing_title_empty_and_ignored_other_abstracts(self):
        source=article(sections=["  "])
        source["other_abstracts"]=[{"sections":[{"text":"Not a fallback"}]}]
        rows=article_windows(source,Tokenizer())
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["kind"],"title")
        self.assertIsNone(rows[0]["section_index"])
        self.assertEqual(article_windows(article("",[" "]),Tokenizer()),[])
        rows=article_windows(article("",["Abstract without title"]),Tokenizer())
        self.assertEqual(rows[0]["title_token_count"],0)
        self.assertEqual(rows[0]["body_token_count"],3)

    def test_config_and_offsets_validation(self):
        for kwargs in ({"body_tokens":0},{"overlap":384},{"overlap":-1},{"max_input_tokens":513},{"overlap":True}):
            with self.assertRaises(WindowError): WindowConfig(**kwargs)
        tokenizer=Tokenizer()
        def bad(text,**kwargs):
            result=tokenizer(text,**kwargs)
            if "offset_mapping" in result: result["offset_mapping"]=[(0,0)]*len(result["input_ids"])
            return result
        fake=type('BadTokenizer',(),{'cls_token_id':101,'sep_token_id':102,
            'num_special_tokens_to_add':lambda self,pair=False:2,'__call__':lambda self,text,**kw:bad(text,**kw)})()
        with self.assertRaisesRegex(WindowError,"offsets"):
            article_windows(article(sections=["some text"]),fake)

    def test_artifact_determinism_offline_replay_corruption_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            synthetic_inputs(root)
            tokenizer=Tokenizer()
            with patch("socket.socket",side_effect=AssertionError("offline")):
                a=build_windows(root/'parsed',tokenizer,{"test":"synthetic"},root/'a')
                b=build_windows(root/'parsed',tokenizer,{"test":"synthetic"},root/'b')
                manifest,rows=load_windows(root/'a')
            self.assertEqual(a["artifact_id"],b["artifact_id"])
            self.assertEqual(len(rows),3)
            self.assertEqual(manifest["title_only_windows"],3)
            with self.assertRaises(FileExistsError):
                build_windows(root/'parsed',tokenizer,{},root/'a')
            path=root/'a/windows.jsonl';path.write_bytes(path.read_bytes()+b"{}\n")
            with self.assertRaisesRegex(WindowError,"checksum"):load_windows(root/'a')

    def test_inspect_cli_needs_no_transformers_or_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);synthetic_inputs(root)
            build_windows(root/'parsed',Tokenizer(),{},root/'windows')
            script=Path(__file__).resolve().parents[1]/'scripts/prepare_windows.py'
            stdout=io.StringIO()
            with patch.object(sys,'argv',[str(script),'inspect',str(root/'windows')]),redirect_stdout(stdout):
                with patch("socket.socket",side_effect=AssertionError("offline")):
                    with self.assertRaises(SystemExit) as result:runpy.run_path(str(script),run_name='__main__')
            self.assertEqual(result.exception.code,0)
            self.assertEqual(json.loads(stdout.getvalue())["window_count"],3)


if __name__=='__main__':unittest.main()
