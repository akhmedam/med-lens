# PubMed ingestion check

2026-10-19. Query: `depression AND cognitive behavioral therapy`.

| Check | Result |
|---|---:|
| Matches reported by PubMed | 23,364 |
| Selected / fetched PMID records | 30 / 30 |
| ESearch / EFetch requests | 1 / 3 |
| Request retries | 0 |
| Raw response bytes | 589,285 |
| Download and validation wall time | 4.94 s |

Command:

```sh
python scripts/pubmed_snapshot.py fetch \
  --query 'depression AND cognitive behavioral therapy' \
  --limit 30 --batch-size 10 \
  --output data/snapshots/2026-10-19-depression-cbt-001
python scripts/pubmed_snapshot.py inspect data/snapshots/2026-10-19-depression-cbt-001
```

All requested primary PMIDs were present. EFetch order is not assumed to match
ESearch order; the manifest preserves the original candidate ranking.

PubMed applied Automatic Term Mapping. The manifest keeps both the submitted
query and PubMed's `query_translation`. No custom query expansion, date filter,
language filter, or publication-type filter was applied.

The [manifest](pubmed-ingestion-manifest.json) records timestamps, parameters,
ordered PMIDs, byte counts, SHA-256 hashes, and attempt counts. Raw JSON/XML stay
in the local snapshot directory and are not committed. Exact offline replay
requires those local files; a new live search may return different results.

The client uses a 20 s socket timeout, at most three attempts, and a sequential
2.5 requests/s limit. It retries HTTP 429/500/502/503/504 and transport failures.
`Retry-After` takes precedence over exponential backoff; an excessively long
cooldown fails the job instead of retrying too early. The limiter is local to one
client, not an IP-wide coordinator.

Raw HTTP-success bodies are saved before JSON/XML validation. Partial or malformed
results mark the snapshot as failed. Interrupted jobs can remain running; the
loader accepts only complete snapshots and verifies their hashes and PMID sets.
This is integrity checking, not a cryptographic authenticity guarantee.

Tests cover retries, timeout, rate limiting, missing and duplicate records,
corruption, and the inspect command with socket creation and DNS disabled.
No article metadata parser or relevance evaluation is included in this check.

Sources: [NCBI usage rules](https://eutilities.github.io/site/API_Key/usageandkey/),
[E-Utilities reference](https://eutilities.github.io/site/Reference_Guide/a_reference/).
