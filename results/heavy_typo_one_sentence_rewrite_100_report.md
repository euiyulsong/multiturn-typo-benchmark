# Heavy-typo retrieval with one-sentence query refinement

## What was compared

We used the same 100 frozen multi-turn cases (25 each from DMV, SSA, VA, and StudentAid), each once clean and once with heavy synthetic typos: **100 matched clean/typo pairs / 200 evaluations**. Heavy typo generation matches [`heavy_typo_router_rewrite_100_report.md`](heavy_typo_router_rewrite_100_report.md): one eligible word per user turn, including the final turn, receives three sequential edits chosen from transpose, delete, insert, and replace.

This run changes the search query construction:

1. A query-refinement call receives the multi-turn context and final question and returns exactly one concise standalone sentence.
2. The router selects tool(s), but **the router's own query is not used for retrieval**. The one-sentence refined query is the exact query searched.
3. Search is retrieval-only: no answer generation and no ReAct follow-up.

All prompts are typo-aware. There is no separate spelling-only correction stage; the query refiner resolves the conversation into one sentence and can infer/correct typoed words, then the tool router selects the domain.

## Exact prompt behavior

The query refiner uses the benchmark rewrite instructions with a stricter output requirement:

> Rewrite the latest user question into a concise standalone search query. Use dialogue context only to resolve references and ellipsis. Preserve intent and details. Do not select tools, answer the question, or introduce facts. Return JSON with one string field named query. The dialogue may contain spelling mistakes or typos. Infer intended words from context and correct them in the standalone query without changing meaning. Return exactly one concise sentence. Do not include a list, preamble, or multiple alternative queries.

The router then uses the standard typo-aware tool-selection prompt (tool descriptions abbreviated here):

> You are a multi-tool search router. Pick one or more useful tools to answer the final user question. Use full dialogue for context. Select no more than two, deduplicate, and order by importance. Search tools: DMV, SSA, VA, and StudentAid domain search. Write a focused standalone retrieval query resolving references only as needed. Return JSON with tools and query. Never use dataset labels. The input may contain spelling errors or typos: infer the intended words from dialogue context, and produce a corrected, focused standalone retrieval query without changing intent.

The router still returns a query because it shares the normal schema, but the retrieved results use the **query-refiner's one-sentence output**. This isolates the effect of making the search string depend on a concise whole-dialogue refinement rather than the router's own query formulation.

## Clean vs. heavy typo results

Recall@5 measures whether gold documents or evidence spans appear in the top five; MRR@10 rewards ranking a gold document near the top ten. Delta is paired `typo − clean`; W/T/L counts cases with higher/equal/lower typo values.

| Metric | Clean | Heavy typo | Paired delta | W / T / L |
|---|---:|---:|---:|---:|
| Router tool-set EM | 0.910 | 0.910 | 0.000 | 4 / 92 / 4 |
| BM25 document Recall@5 | 0.660 | 0.620 | −0.040 | 5 / 86 / 9 |
| BM25 span Recall@5 | 0.533 | 0.503 | −0.030 | 3 / 91 / 6 |
| BM25 MRR@10 | 0.511 | 0.481 | −0.030 | 15 / 68 / 17 |
| Dense document Recall@5 | 0.530 | 0.530 | 0.000 | 4 / 92 / 4 |
| Dense span Recall@5 | 0.423 | 0.400 | −0.023 | 4 / 90 / 6 |
| Dense MRR@10 | 0.463 | 0.443 | −0.020 | 11 / 77 / 12 |
| RRF document Recall@5 | 0.690 | 0.610 | **−0.080** | 1 / 90 / 9 |
| RRF span Recall@5 | 0.583 | 0.510 | **−0.073** | 4 / 84 / 12 |
| RRF MRR@10 | 0.536 | 0.495 | **−0.041** | 12 / 67 / 21 |
| Cross-encoder document Recall@5 | 0.730 | 0.680 | −0.050 | 2 / 91 / 7 |
| Cross-encoder span Recall@5 | 0.655 | 0.615 | −0.040 | 5 / 85 / 10 |
| Cross-encoder MRR@10 | 0.581 | 0.584 | +0.003 | 12 / 73 / 15 |

In this pipeline, the heavier typos lower most retrieval measures. The largest drops are RRF document Recall@5 (−0.08), RRF span Recall@5 (−0.073), and cross-encoder document Recall@5 (−0.05). Router EM stays the same. Cross-encoder MRR is essentially flat, showing again that document inclusion, evidence-span coverage, and rank can move differently.

## Against the existing router-query pipeline

Both runs use the same cases, model, typo edits, two model calls per case, staged architecture, and typo-aware instructions. The changed factor is which query is sent to search:

| Typo condition metric | Standard staged rewrite+router query | One-sentence refiner query | Paired change (one-sentence − standard) |
|---|---:|---:|---:|
| Router tool-set EM | 0.940 | 0.910 | −0.030 |
| BM25 document Recall@5 | 0.690 | 0.620 | −0.070 |
| BM25 span Recall@5 | 0.553 | 0.503 | −0.050 |
| BM25 MRR@10 | 0.544 | 0.481 | −0.063 |
| Dense document Recall@5 | 0.550 | 0.530 | −0.020 |
| Dense span Recall@5 | 0.457 | 0.400 | −0.057 |
| Dense MRR@10 | 0.463 | 0.443 | −0.020 |
| RRF document Recall@5 | 0.700 | 0.610 | **−0.090** |
| RRF span Recall@5 | 0.587 | 0.510 | −0.077 |
| RRF MRR@10 | 0.569 | 0.495 | −0.073 |
| Cross-encoder document Recall@5 | 0.720 | 0.680 | −0.040 |
| Cross-encoder span Recall@5 | 0.667 | 0.615 | −0.052 |
| Cross-encoder MRR@10 | 0.619 | 0.584 | −0.035 |

The one-sentence refinement did **not** improve retrieval in this comparison; the standard rewrite+router-query pipeline scored higher on nearly all reported typo-case retrieval metrics. A likely explanation is that the standard path lets the router formulate a search-specific query after seeing the rewritten question, while the strict one-sentence path locks search to the separate refiner output. The current data do not show that concise rewriting itself is harmful in every setup; they show that **using only that sentence as the final query** underperformed the router's query in this experiment.

## Fairness and limitations

- Same 100 fixed case IDs and same clean/heavy-typo variants as the standard router+rewrite run; 100 matched pairs in each run.
- The comparison holds model, typo-aware instructions, correction strategy (`raw`), staged architecture, retrieval index, tools, and gold labels fixed. The search-query source is the intended changed factor.
- The separate query-refiner and router are both LLM calls in each setup; API call count is matched. The router's query output is ignored only in this one-sentence-search variant.
- A one-sentence query can omit a needed facet from a multi-turn dialogue. This is a deliberate test of the user's requested “one sentence query” approach, not a claim that all query refinement should be single-sentence.
- All results are retrieval-only; no generated answer or user-facing correctness was evaluated.
- One output per case; no confidence intervals/significance tests. Interpret the consistent negative means as a strong practical signal in this sample, not universal proof.
- Batch size and max concurrent case evaluations were both set to 100. Run recorded 400 LLM calls and 154,554 tokens.
- Raw dialogue rows remain local in ignored `results-heavy-typos-one-sentence-100/`; only aggregates are reported here.
