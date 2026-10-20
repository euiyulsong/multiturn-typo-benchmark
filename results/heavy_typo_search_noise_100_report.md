# Search noise under heavier typo corruption (no router)

## Experiment setup

This is a retrieval-only diagnostic with the LLM router skipped. Each of the **same 100 frozen multi-turn dialogue cases** was searched once with unchanged source input (`clean`) and once with stronger synthetic corruption (`heavy typo`): 100 complete matched pairs / 200 retrievals. The index is fixed to the case's gold domain to isolate retriever sensitivity from routing.

All selected cases are genuinely multi-turn: each has **1–7 prior user turns**, followed by the target question (2–8 user turns total; mean 4.56 including the target). The existing agent turns are interleaved and preserved verbatim.

Compared with the earlier diverse typo test, where each eligible user turn had one edited word with one character operation, this stress test applies **three sequential character edits to one selected eligible word in every user turn**, including the final question. Each edit independently chooses transpose, delete, insert, or replace with equal probability. The 100 typo cases contain 408 corrupted words (4.08 per case) and **1,224 character operations** (12.24 per case). Agent turns, gold answer, documents, and support spans are unchanged.

The synthetic corruption is intentionally stronger than ordinary typo noise. Many results below should be interpreted as stress-test sensitivity, not typical user behavior.

## Paired retrieval results

Recall@5 is the share of gold documents/passages in the top five. MRR@10 rewards ranking the first gold document near the top ten. Delta is mean per-case `heavy typo − clean`; W/T/L counts cases with higher/equal/lower metric values under heavy typo.

| Metric | Clean | Heavy typo | Paired delta | W / T / L |
|---|---:|---:|---:|---:|
| BM25 document Recall@5 | 0.540 | 0.550 | +0.010 | 4 / 93 / 3 |
| BM25 span Recall@5 | 0.420 | 0.413 | −0.007 | 2 / 95 / 3 |
| BM25 MRR@10 | 0.399 | 0.353 | **−0.046** | 11 / 67 / 22 |
| Dense document Recall@5 | 0.600 | 0.580 | −0.020 | 2 / 94 / 4 |
| Dense span Recall@5 | 0.413 | 0.373 | **−0.040** | 3 / 88 / 9 |
| Dense MRR@10 | 0.475 | 0.439 | **−0.036** | 8 / 76 / 16 |
| RRF document Recall@5 | 0.640 | 0.680 | +0.040 | 7 / 90 / 3 |
| RRF span Recall@5 | 0.530 | 0.463 | **−0.067** | 3 / 87 / 10 |
| RRF MRR@10 | 0.518 | 0.510 | −0.008 | 17 / 70 / 13 |
| Cross-encoder document Recall@5 | 0.620 | 0.670 | +0.050 | 6 / 93 / 1 |
| Cross-encoder span Recall@5 | 0.468 | 0.453 | −0.015 | 3 / 93 / 4 |
| Cross-encoder MRR@10 | 0.509 | 0.499 | −0.010 | 15 / 69 / 16 |

The largest decreases are **RRF span Recall@5 (−0.067)**, **Dense MRR@10 (−0.036)**, and **Dense span Recall@5 (−0.040)**. BM25 document Recall@5 stays almost flat, and RRF/cross-encoder document Recall@5 rise, despite lower RRF span coverage and mostly lower MRR. This means finding a relevant document in the top five does not guarantee finding/ranking the annotated supporting passage well.

## Increasing corruption intensity

Three retrieval-only no-router settings were compared on the same cases and gold-domain index:

| Typo setting | Corruption per eligible user turn | BM25 MRR@10 delta | Dense MRR@10 delta | RRF span Recall@5 delta | CE span Recall@5 delta |
|---|---|---:|---:|---:|---:|
| One operation per selected word, broad types | one selected word × 1 operation | −0.045 | −0.023 | −0.042 | −0.025 |
| Heavy: three operations per selected word | one selected word × 3 operations | −0.046 | −0.036 | −0.067 | −0.015 |

The stronger perturbation worsens dense MRR and RRF span recall further, while BM25 MRR is almost unchanged. Cross-encoder span recall decline is smaller in the heavy test than in the one-operation test. Effects are not perfectly monotonic across all metrics: interaction between operation type, tokenization, ranking, and small-case ties can produce non-monotonic results.

## Interpretation

- This test deliberately removes routing and query rewriting: **no LLM prompt or model call** was used. It quantifies raw retrieval noise when the correct domain is known.
- The strongest signal is loss in rank/span quality rather than a consistent loss in top-five document recall. Relevant documents often remain present, but the exact support chunk or its rank degrades for some retrievers.
- Three character edits to one word is harsh. It helps stress the lexical and semantic retrieval stack but should not be treated as an estimate of ordinary user typo frequency.
- The contrast with typo-aware router runs is expected: a router that receives context can formulate a clean standalone query; this diagnostic feeds the raw multi-turn text directly into retrieval.

## Exact conditions and reproducibility

- `clean`: source history and final user question unchanged; existing natural errors remain.
- `heavy typo`: deterministic RNG from case ID / turn / seed `20261020`; one eligible word per user turn receives three sequential random operations from transpose, delete, insert, replace.
- Gold domain index fixed from annotation; search query is all original/perturbed turns concatenated in order; no tool router, query rewrite, spell correction, answer generation, or ReAct.
- 100 frozen cases; 25/domain; 200 retrieval conditions total.
- Batch size 50; concurrency 50 (LLM concurrency is unused here; all LLM calls/tokens are zero).
- Outputs are local in ignored `results-heavy-typos-no-router-100/`; no dialogue rows were uploaded.
- No confidence intervals or significance tests; means and paired W/T/L counts are descriptive.
