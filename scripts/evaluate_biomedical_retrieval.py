#!/usr/bin/env python3
"""Evaluate saved biomedical rankings against an explicitly assistant-origin pool.

No network, inference, annotation mutation, or held-out input is used. This is
an exploratory development diagnostic, not independent human validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


class EvaluationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvaluationError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_bytes(path):
    require(path.is_file() and not path.is_symlink(), f"Expected regular file: {path}")
    return path.read_bytes()


def sha(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def parse(body):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise EvaluationError(f"Nonfinite JSON value: {value}")

    value = json.loads(body, object_pairs_hook=pairs, parse_constant=constant)
    canonical(value)
    return value


def read(path):
    return parse(read_bytes(path))


def child(directory, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "Invalid manifest path")
    parts = Path(relative).parts
    require(not Path(relative).is_absolute() and all(x not in ("", ".", "..") for x in parts),
            "Unsafe manifest path")
    path = directory / relative
    require(path.resolve().is_relative_to(directory.resolve()), "Path escapes artifact directory")
    return path


def check_inventory(directory, files):
    for relative, expected in files.items():
        body = read_bytes(child(directory, relative))
        require(len(body) == expected["bytes"] and hashlib.sha256(body).hexdigest() == expected["sha256"],
                f"Checksum mismatch: {directory.name}/{relative}")
    return len(files)


def metrics(ranking, candidates, grades):
    k = 10
    require(len(set(ranking)) == len(ranking) and len(set(candidates)) == len(candidates),
            "Duplicate PMID in ranking/candidates")
    require(set(ranking) <= set(candidates), "Ranking contains a noncandidate PMID")
    require(set(grades) <= set(candidates), "Pool contains a noncandidate PMID")
    require(all(type(x) is int and x in (0, 1, 2) for x in grades.values()),
            "This diagnostic requires rated 0/1/2 labels; unknown is not zero")
    top = ranking[:k]
    require(set(top) <= set(grades), "Unjudged top-ranked PMID; refusing to score it as zero")
    gains = [2 ** grades[p] - 1 for p in top]
    ideal = sorted((2 ** grade - 1 for grade in grades.values()), reverse=True)[:k]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    result = {"ndcg@10": dcg / idcg if idcg else None,
              "dcg@10": dcg, "pooled_idcg@10": idcg,
              "returned_count": len(top), "judged_returned_count": len(top),
              "unjudged_returned_count": 0,
              "judged_coverage@10": 1.0 if top else None,
              "candidate_judged_count": len(grades),
              "candidate_count": len(candidates),
              "candidate_judged_coverage": len(grades) / len(candidates) if candidates else None}
    for threshold, name in ((1, "useful"), (2, "direct")):
        relevant = {p for p, grade in grades.items() if grade >= threshold}
        hits = sum(p in relevant for p in top)
        result[name] = {
            "grade_threshold": threshold, "hits@10": hits,
            "known_relevant_in_pool": len(relevant),
            "precision@10": hits / k,
            "precision_among_returned": hits / len(top) if top else None,
            "pool_recall@10": hits / len(relevant) if relevant else None,
            "mrr@10": next((1 / rank for rank, p in enumerate(top, 1) if p in relevant), 0.0)
            if relevant else None,
        }
    return result


def macro(values):
    defined = [v for v in values if v is not None]
    return {"mean": sum(defined) / len(defined) if defined else None,
            "defined_queries": len(defined), "total_queries": len(values)}


def bound_grades(items_list, judgments, qrels_body):
    """Reject labels bound to any other question/article/text-version pair."""
    items = {(i["query_id"], i["pmid"]): i for i in items_list}
    require(len(items) == len(items_list), "Duplicate packet pair")
    grades = {}
    for row in judgments:
        key = row["query_id"], row["pmid"]
        require(key in items and key not in grades, "Unknown/duplicate assistant judgment")
        item = items[key]
        require(all(row[k] == item[k] for k in ("item_id", "text_version_id", "item_sha256")),
                "Assistant label/source binding mismatch")
        require(row["state"] == "rated" and type(row["relevance"]) is int
                and row["relevance"] in (0, 1, 2), "Incomplete/unknown assistant judgment")
        grades[key] = row["relevance"]
    qrels = {}
    for line in qrels_body.splitlines():
        row = parse(line)
        key = row["query_id"], row["pmid"]
        require(key not in qrels and type(row["relevance"]) is int, "Invalid/duplicate qrels row")
        qrels[key] = row["relevance"]
    require(qrels == grades and set(grades) == set(items), "Qrels/annotations/pool differ")
    return items, grades


def evaluate(sources, package_dir, annotation_dir):
    manifest = read(package_dir / "manifest.json")
    package = read(package_dir / "package.json")
    annotation = read(annotation_dir / "assistant-annotations.json")
    provenance = read(annotation_dir / "assistant-qrels.jsonl.provenance.json")
    inventory = read(annotation_dir / "artifact-files.json")
    packet_files = check_inventory(package_dir, manifest["files"])
    assistant_files = check_inventory(annotation_dir, inventory["files"])
    require(manifest["manifest_id"] == digest({k: v for k, v in manifest.items() if k != "manifest_id"}),
            "Packet manifest identity mismatch")
    require(manifest["package_id"] == package["package_id"] == digest(manifest["identity"]),
            "Packet identity mismatch")
    require(package["split"] == "dev" and package["heldout_used"] is False,
            "Only development packets are accepted")
    require(package["items_sha256"] == digest(package["items"])
            and package["source_bindings_sha256"] == digest(manifest["sources"]),
            "Packet source/item digest mismatch")
    require(annotation["artifact_id"] == digest({k: v for k, v in annotation.items() if k != "artifact_id"}),
            "Assistant annotation identity mismatch")
    for obj in (annotation, provenance):
        require(obj.get("judgments_origin") == "assistant" and obj.get("human_authored") is False
                and obj.get("independent_human_reference") is False,
                "This separate diagnostic accepts assistant-origin labels only")
        require(obj["package_id"] == package["package_id"], "Annotation/package mismatch")
    require(annotation["package_file_sha256"] == sha(package_dir / "package.json"),
            "Assistant labels refer to different package bytes")
    require(annotation["plan_sha256"] == package["plan_sha256"] == sha(package_dir / "plan.json"),
            "Frozen plan identity mismatch")
    require(provenance["qrels_sha256"] == sha(annotation_dir / "assistant-qrels.jsonl")
            and provenance["annotation_json_sha256"] == sha(annotation_dir / "assistant-annotations.json")
            and provenance["source_annotation_artifact_id"] == annotation["artifact_id"],
            "Assistant qrels provenance mismatch")
    items, grades = bound_grades(package["items"], annotation["judgments"],
                                read_bytes(annotation_dir / "assistant-qrels.jsonl"))

    # Verify frozen copies and supplied original source files without relying on
    # absolute paths from another checkout. The archived plan is the same bytes.
    original_checks = 0
    for original, expected in manifest["source_original_sha256"].items():
        matches = [(row["query_id"], original.split("/" + row["query_id"] + "/", 1)[1])
                   for row in manifest["sources"] if "/" + row["query_id"] + "/" in original]
        if not matches:
            require(original.endswith("/benchmarks/biomedical-dev-v1/plan.json"), "Unknown original source path")
            path = package_dir / "plan.json"
        else:
            require(len(matches) == 1, "Ambiguous original source path")
            qid, suffix = matches[0]
            path = child(sources, qid + "/" + suffix)
        require(sha(path) == expected, f"Original source checksum mismatch: {path}")
        original_checks += 1

    cases = []
    for binding in manifest["sources"]:
        qid = binding["query_id"]
        case_dir = child(sources, qid)
        run = read(case_dir / "run.json")
        snapshot = read(case_dir / "snapshot/manifest.json")
        search = read(case_dir / "snapshot/search.json")
        candidates = binding["candidate_pmids"]
        require(sha(case_dir / "run.json") == binding["run_sha256"], "Run checksum mismatch")
        require(run["generate"] is False and run["status"] == "complete"
                and run["runtime_load_count"] == 0, "Source is not completed search-only run")
        require(snapshot["pmids"] == search["esearchresult"]["idlist"] == candidates,
                "PubMed candidate order mismatch")
        require(binding["methods"]["pubmed"] == candidates[:10]
                and binding["methods"]["dense"] == [x["pmid"] for x in run["retrieval"]["results"]],
                "Ranking differs from frozen run")
        require(all(run[k] == binding[k] for k in ("dataset_id", "window_artifact_id", "embedding_artifact_id"))
                and run["live"]["snapshot_id"] == snapshot["snapshot_id"] == binding["snapshot_id"],
                "Source artifact identity mismatch")
        articles = {}
        for line in read_bytes(case_dir / "dataset/articles.jsonl").splitlines():
            article = parse(line)
            require(article["pmid"] not in articles, "Duplicate source article")
            articles[article["pmid"]] = article
        pool_grades = {p: grade for (q, p), grade in grades.items() if q == qid}
        require(set(pool_grades) == set(binding["methods"]["pubmed"]) | set(binding["methods"]["dense"]),
                "Judged pool is not the exact union of methods")
        for pmid in pool_grades:
            item = items[qid, pmid]
            article = articles[pmid]
            require(item["article_sha256"] == digest(article)
                    and item["text_version_id"] == article["text_version"]["text_version_id"],
                    "Pooled label refers to another article/text version")
            require(item["item_sha256"] == digest({k: v for k, v in item.items() if k != "item_sha256"}),
                    "Item identity mismatch")
        cases.append({
            "query_id": qid, "question": run["question"],
            "source": {**{k: binding[k] for k in ("snapshot_id", "dataset_id", "window_artifact_id",
                                                "embedding_artifact_id", "run_sha256")},
                       "submitted_query": run["live"]["query_plan"]["submitted_query"],
                       "total_pubmed_matches": run["live"]["total_pubmed_matches"],
                       "retrieved_finished_at": run["live"]["retrieved_finished_at"]},
            "candidate_count": len(candidates), "judged_pool_count": len(pool_grades),
            "pool_grade_counts": {str(g): sum(v == g for v in pool_grades.values()) for g in (0, 1, 2)},
            "ranking_encoder": run["retrieval"]["encoder"],
            "methods": {method: {"ranking": ranking, "grades": [pool_grades[p] for p in ranking],
                                 "metrics": metrics(ranking, candidates, pool_grades)}
                        for method, ranking in binding["methods"].items()},
        })
    aggregations = {}
    for method in ("pubmed", "dense"):
        rows = [case["methods"][method]["metrics"] for case in cases]
        aggregations[method] = {"ndcg@10": macro([r["ndcg@10"] for r in rows])}
        for threshold in ("useful", "direct"):
            aggregations[method][threshold] = {
                key: macro([r[threshold][key] for r in rows])
                for key in ("precision@10", "precision_among_returned", "pool_recall@10", "mrr@10")}
    return {
        "schema_version": 1, "kind": "exploratory-biomedical-retrieval-assessment",
        "split": "dev", "heldout_used": False, "generation_evaluated": False,
        "judgments_origin": "assistant", "human_authored": False,
        "independent_human_reference": False, "annotation_model_identity": annotation["actual_model_identity"],
        "package_id": package["package_id"], "annotation_artifact_id": annotation["artifact_id"],
        "plan_sha256": package["plan_sha256"], "qrels_sha256": provenance["qrels_sha256"],
        "annotation_json_sha256": provenance["annotation_json_sha256"],
        "evaluator_sha256": sha(Path(__file__)),
        "metric_policy": {
            "k": 10, "graded_gain": "2**grade - 1", "discount": "log2(rank + 1)",
            "ndcg_denominator": "IDCG from the judged union pool, not all PubMed literature",
            "useful_threshold": "grade >= 1", "direct_threshold": "grade == 2",
            "precision_at_10_denominator": "fixed 10, including short lists",
            "precision_among_returned_denominator": "actual returned length",
            "pool_recall_denominator": "relevant judged pairs in the union pool only",
            "empty_relevant_pool": "nDCG, recall and MRR are null; precision remains defined",
            "missing_top_judgments": "fail rather than treat unjudged as zero",
            "macro_policy": "unweighted mean of defined queries, with counts",
            "limits": ["Six development cases; no independent clinical validation or significance claim",
                       "Pool-limited recall does not measure candidate acquisition recall",
                       "Unjudged candidates outside both top-10 lists remain unjudged",
                       "A pool with no known relevant article cannot establish absence of an effect"],
        },
        "verification": {"status": "passed", "packet_files_checked": packet_files,
                         "assistant_files_checked": assistant_files,
                         "original_source_files_checked": original_checks,
                         "label_source_pairs_checked": len(items)},
        "coverage": {"question_count": len(cases), "judged_pairs": len(items),
                     "candidate_pairs": sum(c["candidate_count"] for c in cases),
                     "judged_candidate_fraction_micro": len(items) / sum(c["candidate_count"] for c in cases),
                     "all_returned_top10_pairs_judged": True},
        "aggregate": aggregations, "cases": cases,
    }


def self_check():
    r = metrics(["b", "a"], ["a", "b", "c"], {"a": 2, "b": 1, "c": 0})
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    require(math.isclose(r["ndcg@10"], expected, rel_tol=1e-12), "nDCG self-check failed")
    require(r["useful"]["precision@10"] == 0.2 and r["useful"]["pool_recall@10"] == 1,
            "Useful precision/recall self-check failed")
    require(r["direct"]["precision@10"] == 0.1 and r["direct"]["pool_recall@10"] == 1,
            "Direct precision/recall self-check failed")
    r = metrics(["a"], ["a"], {"a": 0})
    require(r["ndcg@10"] is None and r["useful"]["pool_recall@10"] is None
            and r["useful"]["precision@10"] == 0, "Empty-relevance self-check failed")
    for ranking, candidates, grades in [(["a", "a"], ["a"], {"a": 2}),
                                        (["a"], ["a"], {}),
                                        (["a"], ["a"], {"a": None}),
                                        (["b"], ["a"], {"a": 1})]:
        try:
            metrics(ranking, candidates, grades)
        except EvaluationError:
            continue
        raise EvaluationError("Invalid-input self-check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--annotation-package", type=Path, required=True)
    parser.add_argument("--assistant-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    self_check()
    require(not args.output.exists(), "Output exists; preserve prior reports")
    result = evaluate(args.sources, args.annotation_package, args.assistant_annotations)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "verification": result["verification"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
