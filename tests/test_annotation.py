"""Blinded pools retain candidate provenance and never fabricate human labels."""

import json
import tempfile
import unittest
from pathlib import Path

from medlens.evaluation.annotation import build_packet, export_qrels
from medlens.evaluation.formats import EvaluationError
from test_evaluation import synthetic_inputs


class AnnotationTests(unittest.TestCase):
    def packet(self, root, hostile=False):
        queries, _ = synthetic_inputs(root)
        if hostile:
            rows=[json.loads(line) for line in queries.read_text().splitlines()]
            rows[0]["text"] = '</script><img src=x onerror="alert(1)">'
            queries.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        plan=root/'plan.json'
        plan.write_text(json.dumps({'queries':[{'query_id':qid,'relevance_intent':'Synthetic fixture only'} for qid in ('q1','q2')]}))
        manifest=build_packet(queries,plan,root/'packet',depth=2)
        return queries,plan,manifest

    def test_union_dedup_blinding_provenance_and_stable_rebuild(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            queries,plan,first=self.packet(root)
            second=build_packet(queries,plan,root/'second',depth=2)
            self.assertEqual(first['pool_id'],second['pool_id'])
            pool=json.loads((root/'packet/pool.json').read_bytes())
            self.assertEqual(set(pool),{'schema_version','pool_id','items'})
            q1=[item for item in pool['items'] if item['query_id']=='q1']
            self.assertEqual({item['pmid'] for item in q1},{'101','102','103'})
            self.assertEqual(len(q1),3)
            for item in pool['items']:
                self.assertNotIn('methods',item)
                self.assertNotIn('rank',item)
                self.assertNotIn('score',item)
                self.assertTrue(item['text_version_id'])
            with self.assertRaises(FileExistsError):
                build_packet(queries,plan,root/'packet')

    def test_untrusted_source_text_cannot_escape_json_script(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.packet(root,hostile=True)
            html=(root/'packet/index.html').read_text()
            self.assertNotIn('<img src=x',html)
            self.assertIn('\\u003c/script>',html)
            self.assertIn('.textContent=item.query',html)

    def labels(self, root, grade):
        pool=json.loads((root/'packet/pool.json').read_bytes())
        result={'schema_version':1,'pool_id':pool['pool_id'], 'judgments':[
            {'query_id':item['query_id'],'pmid':item['pmid'],'relevance':grade} for item in pool['items']]}
        path=root/'annotations.json'
        path.write_text(json.dumps(result))
        return path,result

    def test_partial_labels_require_opt_in_and_complete_export_keeps_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.packet(root)
            labels,_=self.labels(root,None)
            with self.assertRaisesRegex(EvaluationError,'Incomplete'):
                export_qrels(root/'packet',labels,root/'rejected.jsonl')
            self.assertFalse((root/'rejected.jsonl').exists())
            partial=export_qrels(root/'packet',labels,root/'partial.jsonl',allow_partial=True)
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['judged_count'],0)
            labels,_=self.labels(root,1)
            receipt=export_qrels(root/'packet',labels,root/'complete.jsonl')
            self.assertTrue(receipt['complete'])
            self.assertEqual(receipt,json.loads((root/'complete.jsonl.receipt.json').read_bytes()))
            with self.assertRaises(FileExistsError):
                export_qrels(root/'packet',labels,root/'complete.jsonl')

    def test_wrong_pool_unknown_pairs_duplicates_and_invalid_grades_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.packet(root)
            path,valid=self.labels(root,0)
            variants=[]
            wrong=json.loads(json.dumps(valid));wrong['pool_id']='other';variants.append(wrong)
            wrong=json.loads(json.dumps(valid));wrong['judgments'][0]['pmid']='999';variants.append(wrong)
            wrong=json.loads(json.dumps(valid));wrong['judgments'].append(wrong['judgments'][0]);variants.append(wrong)
            wrong=json.loads(json.dumps(valid));wrong['judgments'][0]['relevance']=True;variants.append(wrong)
            wrong=json.loads(json.dumps(valid));wrong['judgments'].pop();variants.append(wrong)
            for index,bad in enumerate(variants):
                path.write_text(json.dumps(bad))
                with self.assertRaises(EvaluationError):
                    export_qrels(root/'packet',path,root/f'bad-{index}.jsonl')
            self.assertFalse(list(root.glob('bad-*.jsonl')))

    def test_changed_pool_bytes_fail_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.packet(root)
            labels,_=self.labels(root,0)
            with (root/'packet/pool.json').open('ab') as stream:
                stream.write(b' ')
            with self.assertRaisesRegex(EvaluationError,'checksum'):
                export_qrels(root/'packet',labels,root/'changed-qrels.jsonl')


if __name__=='__main__':
    unittest.main()
