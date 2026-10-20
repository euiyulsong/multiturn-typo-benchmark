# Controlled query-refinement ablation under heavy typos

## Question and design

This experiment tests the user's concern directly: **does refining the final question using multi-turn context help if both arms then use the same router → router query → retrieval pipeline?**

Both arms use the same 100 frozen multi-turn cases (25 per domain), heavy typo generation, typo-aware prompt, model, staged architecture, tools, and rankers. Each case has a clean and a heavy-typo evaluation. The controlled factor is whether the router receives the original final question or a version first rewritten using full dialogue context:

1. **Unrefined baseline:** the standard rewrite step receives only the final question (without history) and produces a standalone query; that query is passed into the router, which selects tools and produces the actual search query.
2. **Refined treatment:** the rewrite step receives the full multi-turn dialogue and final question and returns one concise standalone sentence; this refined sentence is passed into the router, which performs the same tool-selection and query-writing step. **Search always uses the router's returned query in both arms.**

Thus this isolates the effect of providing multi-turn dialogue context to the query-refinement step **before the same router/query/search pipeline**, unlike the prior one-sentence-only experiment that searched the refiner's sentence directly. Both arms make one rewrite call and one router call; only the rewrite step's access to dialogue history differs.

## Results

Clean/typo below are the means for the **full-context refinement treatment**. Paired delta is heavy typo minus clean within that treatment. “Refined − baseline” is case-matched on the same condition and compares full-context refinement against the final-question-only rewrite baseline; it measures the effect of giving the rewriter dialogue history under a shared router/search design.

| Metric | Refined clean | Refined heavy typo | Typo − clean | Refined − baseline on typo |
|---|---:|---:|---:|---:|
| Router tool-set EM | 0.920 | 0.950 | +0.030 | +0.030 |
| Router tool-set F1 | 0.947 | 0.963 | +0.017 | +0.010 |
| BM25 document Recall@5 | 0.660 | 0.660 | 0.000 | −0.030 |
| BM25 span Recall@5 | 0.533 | 0.533 | 0.000 | −0.020 |
| BM25 MRR@10 | 0.514 | 0.528 | +0.014 | −0.016 |
| Dense document Recall@5 | 0.550 | 0.570 | +0.020 | +0.020 |
| Dense span Recall@5 | 0.453 | 0.457 | +0.003 | +0.000 |
| Dense MRR@10 | 0.473 | 0.472 | −0.001 | +0.009 |
| RRF document Recall@5 | 0.680 | 0.700 | +0.020 | +0.000 |
| RRF span Recall@5 | 0.562 | 0.590 | +0.028 | +0.003 |
| RRF MRR@10 | 0.519 | 0.548 | +0.029 | −0.021 |
| Cross-encoder document Recall@5 | 0.720 | 0.730 | +0.010 | +0.010 |
| Cross-encoder span Recall@5 | 0.647 | 0.647 | +0.000 | −0.020 |
| Cross-encoder MRR@10 | 0.612 | 0.618 | +0.006 | −0.001 |

**In this sample, adding full-dialogue context to the query refiner had mixed, mostly modest effects compared with rewriting only the final question.** On typo inputs, router EM and dense document Recall@5 were 0.02 higher; BM25 document Recall@5 was 0.03 lower, RRF document/span Recall@5 were 0.01/0.003 higher, and cross-encoder span Recall@5 was 0.02 lower. Within the full-context refinement arm, clean-to-heavy-typo changes were small for BM25 and cross-encoder recall, while RRF recall/span coverage declined.

This differs from the previous `rewrite_only_search` arm: there, retrieval used the refiner's sentence directly. In this controlled ablation, the router receives the refined sentence and retrieval uses the router's query, exactly as requested for an apples-to-apples pipeline comparison.

## Interpretation

- Giving the rewrite step full dialogue context had **mixed, mostly modest effects** relative to rewriting only the final question. It helped router EM and dense document recall in the typo arm, while other metrics moved in both directions.
- The refined one-sentence query can resolve references and typos, but may also compress or reweight details. The router still sees the full history and produces the final search query.
- Refinement did not change the search pipeline after the router: both arms route and search using router-selected tool(s) and router-generated query. The earlier “search only the refiner's sentence” result is a separate, less controlled formulation and should not be conflated with this test.
- Heavy-typo inputs are deliberately harsh (three character edits to one selected word per eligible user turn). Results characterize this stress set, not typical user misspellings.
- No confidence intervals or significance test; each case is evaluated once per condition. Treat these as paired descriptive evidence, not a universal claim that refinement is harmful.

## Prompts and run record

The baseline query-refinement prompt receives only the final question (without conversation history) and requests a concise standalone query. The treatment prompt additionally receives the full dialogue history, with the requirement to output exactly one concise sentence:

> Rewrite the latest user question into a concise standalone search query. Use dialogue context only to resolve references and ellipsis. Preserve intent and details. Do not select tools, answer the question, or introduce facts. Return JSON with one string field named query. The dialogue may contain spelling mistakes or typos. Infer intended words from context and correct them in the standalone query without changing meaning. Return exactly one concise sentence. Do not include a list, preamble, or multiple alternative queries.

The typo-aware router prompt and subsequent pipeline are unchanged between arms. In both arms, the router receives the full history plus the corresponding rewrite result, chooses up to two domain tools, and produces the query sent to retrieval.

- Same 100 frozen case IDs, 25/domain; 100 clean + 100 heavy typo = 100 matched pairs / 200 evaluations.
- Model: `gpt-6-luna`; strategy `raw`; staged architecture; typo-aware router; retrieval-only.
- Dense-index embedding batch size 100; max concurrent case evaluations 100.
- Recorded model calls: 400; tokens: 155,085.
- Prior unrefined router+rewrite baseline: 400 calls; 149,398 tokens.
- Raw row/detail outputs remain local in ignored `results-heavy-typos-controlled-refine-100/`.
- Dollar cost is omitted because exact billed model/account rates were not independently verified.
