# Heavy typo retrieval with router + query rewriting

## Setup

This run repeats the **same 100 frozen multi-turn cases** (25 each in DMV, SSA, VA, StudentAid) in clean and heavy-typo form: 100 matched pairs / 200 retrieval-only evaluations. It uses the same high-intensity typo construction as [`heavy_typo_search_noise_100_report.md`](heavy_typo_search_noise_100_report.md): one selected eligible word in every user turn receives **three sequential character edits**, chosen from transpose/delete/insert/replace. The 100 typo dialogues contain 408 edited words and 1,224 individual character operations (12.24 operations/case on average).

Unlike the no-router diagnostic, this is the **router + query-rewrite path**:

1. A separate LLM query-rewrite call sees the dialogue context and final question and produces a concise standalone query.
2. The router LLM selects one or more search tools and produces a focused standalone retrieval query. Its prompt is typo-aware and explicitly asks it to infer intended words from context and correct the query without changing intent.
3. The rewritten query is searched using the tools selected by the router. We stop after this first search: no answer generation or ReAct follow-up.

The same router prompt is used for both clean and heavy-typo conditions. No separate spell-correction stage was used (`strategy=raw`). Batch size for the offline dense index was 100, and up to 100 independent case evaluations were allowed concurrently.

## Exact prompts

### Query-rewrite prompt

> Rewrite the latest user question into a concise standalone search query. Use dialogue context only to resolve references and ellipsis. Preserve intent and details. Do not select tools, answer the question, or introduce facts. Return JSON with one string field named query. The dialogue may contain spelling mistakes or typos. Infer intended words from context and correct them in the standalone query without changing meaning.

The runtime instruction has the typo sentence appended when typo-awareness is on, as it was for this run.

### Tool-router prompt

> You are a multi-tool search router. Pick one or more useful tools to answer the final user question. Use full dialogue for context. Select no more than two, deduplicate, and order by importance. Search tools: dmv = driver's license, registration, state motor vehicles; ssa = Social Security, retirement, benefits, disability; va = US Veterans Affairs and veterans benefits; studentaid = federal student aid, FAFSA, grants and loans. Write a focused standalone retrieval query resolving references only as needed. Return JSON with tools and query. Never use dataset labels. The input may contain spelling errors or typos: infer the intended words from dialogue context, and produce a corrected, focused standalone retrieval query without changing intent.

The router receives the full dialogue plus the query rewrite as its question input; strict JSON output contains selected `tools` and `query` fields. Tool selection is measured against the dataset's gold domain tool(s).

## Paired clean vs. heavy-typo results

Recall@5 is the share of gold documents/passages in the top five; MRR@10 rewards ranking a gold document near the top ten. Delta is the mean paired `typo − clean` difference. W/T/L counts per-case higher/equal/lower typo scores.

| Metric | Clean | Heavy typo | Paired delta | W / T / L |
|---|---:|---:|---:|---:|
| Router tool-set EM | 0.920 | 0.940 | +0.020 | 3 / 96 / 1 |
| Router tool-set F1 | 0.947 | 0.953 | +0.007 | 3 / 96 / 1 |
| BM25 document Recall@5 | 0.660 | 0.690 | +0.030 | 8 / 87 / 5 |
| BM25 span Recall@5 | 0.533 | 0.553 | +0.020 | 7 / 88 / 5 |
| BM25 MRR@10 | 0.514 | 0.544 | +0.030 | 18 / 72 / 10 |
| Dense document Recall@5 | 0.550 | 0.550 | 0.000 | 3 / 94 / 3 |
| Dense span Recall@5 | 0.453 | 0.457 | +0.003 | 5 / 90 / 5 |
| Dense MRR@10 | 0.473 | 0.463 | −0.010 | 13 / 76 / 11 |
| RRF document Recall@5 | 0.680 | 0.700 | +0.020 | 7 / 88 / 5 |
| RRF span Recall@5 | 0.562 | 0.587 | +0.025 | 9 / 85 / 6 |
| RRF MRR@10 | 0.519 | 0.569 | +0.049 | 17 / 74 / 9 |
| Cross-encoder document Recall@5 | 0.720 | 0.720 | 0.000 | 2 / 96 / 2 |
| Cross-encoder span Recall@5 | 0.647 | 0.667 | +0.020 | 6 / 90 / 4 |
| Cross-encoder MRR@10 | 0.612 | 0.619 | +0.007 | 13 / 74 / 13 |

In this run, the heavy-typo condition does **not** show a broad search drop after router and query rewriting. Router EM rises slightly (+0.02); BM25, RRF, and cross-encoder rank/span metrics are stable or higher; dense MRR declines slightly (−0.01), with dense Recall@5 unchanged. These deltas are descriptive, mostly with many case-level ties, and do not prove typo inputs improve retrieval.

## Does rewriting mitigate the raw retrieval noise?

Compare heavy-typo deltas with the no-router gold-domain diagnostic on the same cases:

| Metric | No-router raw-dialogue delta | Router + rewrite delta |
|---|---:|---:|
| BM25 document Recall@5 | +0.010 | +0.030 |
| BM25 span Recall@5 | −0.007 | +0.020 |
| BM25 MRR@10 | −0.046 | +0.030 |
| Dense document Recall@5 | −0.020 | 0.000 |
| Dense span Recall@5 | −0.040 | +0.003 |
| Dense MRR@10 | −0.036 | −0.010 |
| RRF document Recall@5 | +0.040 | +0.020 |
| RRF span Recall@5 | −0.067 | +0.025 |
| RRF MRR@10 | −0.008 | +0.049 |
| Cross-encoder document Recall@5 | +0.050 | 0.000 |
| Cross-encoder span Recall@5 | −0.015 | +0.020 |
| Cross-encoder MRR@10 | −0.010 | +0.007 |

The starkest contrast is BM25 MRR@10 (−0.046 without routing/rewrite vs. +0.030 with it) and RRF span Recall@5 (−0.067 vs. +0.025). Dense MRR remains slightly lower in both setups but its decline is smaller with rewriting (−0.036 vs. −0.010). This is consistent with the router/rewrite producing a more normalized standalone query that buffers some spelling noise.

**Caveat:** this is not a randomized causal ablation of rewriting alone. The no-router run uses the gold domain and raw full-dialogue text as the query; the router+rewrite run uses model-selected tools and a rewritten query. Pipeline, query length/form, and selection all change. The comparison is useful diagnostically, but does not isolate the effect of one prompt sentence.

## Fairness and limitations

- Both conditions in this router+rewrite run use the exact same 100 case IDs and the same deterministic heavy-typo edits; each case has one clean and one typo result.
- The data are multi-turn: 1–7 prior user turns plus the target question. Three edits are applied to one selected word in each eligible user turn; AGENT turns and gold labels remain unchanged.
- The typo-aware rewrite and router prompts are enabled in both conditions. This measures their robustness to heavier input corruption, not an unaware model.
- Gold-domain no-router scores are a diagnostic, while router+rewrite scores represent an end-to-end first-search path. Compare them cautiously because they form different search queries and select tools differently.
- Search-only: no generated answer is evaluated, and ReAct is off. Retrieval scores do not imply answer correctness.
- One output per case; paired means and W/T/L only, with no confidence intervals or significance tests.
- Model: `gpt-6-luna`; 100 clean + 100 typo evaluations. Batch size 100, max concurrent evaluations 100.
- Recorded API use: 400 LLM calls (rewrite + router per evaluation), 149,398 tokens (71,569 clean; 77,829 typo). Dollar cost is omitted because exact account/model rates were not independently verified.
- Raw rows/details remain in ignored local `results-heavy-typos-router-rewrite-100/` because they contain dialogue content.
