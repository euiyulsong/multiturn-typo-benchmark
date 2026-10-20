# Controlled query-refinement ablation under heavy typos

> The report initially used a partial-history input in the refinement treatment. The **corrected full-history rerun** is now the primary result below; the earlier runs are retained only for audit.

## Question and design

This experiment tests the user's concern directly: **does refining the final question using multi-turn context help if both arms then use the same router → router query → retrieval pipeline?**

Both arms use the same 100 frozen multi-turn cases (25 per domain), heavy typo generation, typo-aware prompt, model, staged architecture, tools, and rankers. Each case has a clean and a heavy-typo evaluation. In the corrected ablation the only intended factor is whether the rewriter receives only the final question or the complete interleaved dialogue:

1. **Baseline:** the rewriter receives only the final question and produces a standalone query. Then the router receives the full interleaved dialogue and that query, selects tools and produces the search query.
2. **Full-history refinement:** the rewriter receives the complete USER/AGENT dialogue and final question, returning one concise sentence. The router then receives the same full dialogue plus that refined question, selects tools and produces the search query. **Search always uses the router's returned query in both arms.**

Thus this isolates the effect of giving dialogue history to the query refiner **before the same router/query/search pipeline**. Both arms make one rewrite call and one router call; only the rewriter's input context differs.

## Corrected full-history results

Clean/typo columns are means for the full-history treatment. “Refined − baseline on typo” is a case-matched comparison on the same heavy-typo cases. Paired delta is heavy typo minus clean within each setup.

| Metric | Baseline clean | Full-history clean | Baseline typo | Full-history typo | Refined − baseline on typo |
|---|---:|---:|---:|---:|---:|
| Router tool-set EM | 0.920 | 0.960 | 0.940 | 0.970 | +0.030 |
| Router tool-set F1 | 0.947 | 0.967 | 0.953 | 0.977 | +0.023 |
| BM25 document Recall@5 | 0.660 | 0.720 | 0.690 | 0.640 | −0.050 |
| BM25 span Recall@5 | 0.533 | 0.620 | 0.553 | 0.538 | −0.015 |
| BM25 MRR@10 | 0.514 | 0.578 | 0.544 | 0.532 | −0.012 |
| Dense document Recall@5 | 0.550 | 0.580 | 0.550 | 0.530 | −0.020 |
| Dense span Recall@5 | 0.453 | 0.473 | 0.457 | 0.433 | −0.023 |
| Dense MRR@10 | 0.473 | 0.492 | 0.463 | 0.454 | −0.009 |
| RRF document Recall@5 | 0.680 | 0.650 | 0.700 | 0.690 | −0.010 |
| RRF span Recall@5 | 0.562 | 0.547 | 0.587 | 0.617 | +0.030 |
| RRF MRR@10 | 0.519 | 0.566 | 0.569 | 0.517 | −0.051 |
| Cross-encoder document Recall@5 | 0.720 | 0.760 | 0.720 | 0.740 | +0.020 |
| Cross-encoder span Recall@5 | 0.647 | 0.705 | 0.667 | 0.672 | +0.005 |
| Cross-encoder MRR@10 | 0.612 | 0.633 | 0.619 | 0.618 | −0.001 |

On heavy-typo inputs, full-history refinement improved router EM (+0.03), RRF span Recall@5 (+0.03), and cross-encoder document Recall@5 (+0.02) versus baseline. It lowered BM25 document Recall@5 (−0.05), Dense document Recall@5 (−0.02), and RRF MRR@10 (−0.051); most other changes were small. This is mixed, not a general win. Within the refined arm, typo input reduced BM25 and Dense recall and several MRR/span measures, while RRF Recall@5 rose.

This differs from the previous `rewrite_only_search` arm: there, retrieval used the refiner's sentence directly. In this controlled ablation, the router receives the refined sentence and retrieval uses the router's query, exactly as requested for an apples-to-apples pipeline comparison.

## Audit: earlier partial-history implementation

The initial implementation passed user turns only (omitting interleaved AGENT turns) as rewriter context. These values are retained for audit and should not be used as the corrected full-history treatment.

| Metric | Refined clean | Refined heavy typo | Typo − clean | Refined − baseline on typo |
|---|---:|---:|---:|---:|
| Router tool-set EM | 0.960 | 0.970 | +0.010 | +0.030 |
| Router tool-set F1 | 0.967 | 0.977 | +0.010 | +0.023 |
| BM25 document Recall@5 | 0.720 | 0.640 | −0.080 | −0.050 |
| BM25 span Recall@5 | 0.620 | 0.538 | −0.082 | −0.015 |
| BM25 MRR@10 | 0.578 | 0.532 | −0.046 | −0.012 |
| Dense document Recall@5 | 0.580 | 0.530 | −0.050 | −0.020 |
| Dense span Recall@5 | 0.473 | 0.433 | −0.040 | −0.023 |
| Dense MRR@10 | 0.492 | 0.454 | −0.038 | −0.009 |
| RRF document Recall@5 | 0.650 | 0.690 | +0.040 | −0.010 |
| RRF span Recall@5 | 0.547 | 0.617 | +0.070 | +0.030 |
| RRF MRR@10 | 0.566 | 0.517 | −0.049 | −0.051 |
| Cross-encoder document Recall@5 | 0.760 | 0.740 | −0.020 | +0.020 |
| Cross-encoder span Recall@5 | 0.705 | 0.672 | −0.033 | +0.005 |
| Cross-encoder MRR@10 | 0.633 | 0.618 | −0.015 | −0.001 |

This earlier partial-history run is superseded by the corrected results above.

## Interpretation

- Giving the rewrite step dialogue context had **mixed effects** relative to rewriting only the final question. In the corrected run it improved typo-case routing EM and RRF span recall, while BM25/Dense/RRF rank or document-recall measures were mixed or lower.
- The refined one-sentence query can resolve references and typos, but may also compress or reweight details. The router still sees the full history and produces the final search query.
- Refinement did not change the search pipeline after the router: both arms route and search using router-selected tool(s) and router-generated query. The earlier “search only the refiner's sentence” result is a separate, less controlled formulation and should not be conflated with this test.
- Heavy-typo inputs are deliberately harsh (three character edits to one selected word per eligible user turn). Results characterize this stress set, not typical user misspellings.
- No confidence intervals or significance test; each case is evaluated once per condition. Treat these as paired descriptive evidence, not a universal claim that refinement is harmful.

## Prompts and run record

The baseline query-refinement prompt receives only the final question (without conversation history) and requests a concise standalone query. The corrected treatment receives the full interleaved USER/AGENT history plus the final question and must output exactly one concise sentence:

> Rewrite the latest user question into a concise standalone search query. Use dialogue context only to resolve references and ellipsis. Preserve intent and details. Do not select tools, answer the question, or introduce facts. Return JSON with one string field named query. The dialogue may contain spelling mistakes or typos. Infer intended words from context and correct them in the standalone query without changing meaning. Return exactly one concise sentence. Do not include a list, preamble, or multiple alternative queries.

The typo-aware router prompt and subsequent pipeline are unchanged between arms. In both arms, the router receives the full history plus the corresponding rewrite result, chooses up to two domain tools, and produces the query sent to retrieval.

- Same 100 frozen case IDs, 25/domain; 100 clean + 100 heavy typo = 100 matched pairs / 200 evaluations.
- Model: `gpt-6-luna`; strategy `raw`; staged architecture; typo-aware router; retrieval-only.
- Dense-index embedding batch size 100; max concurrent case evaluations 100.
- Recorded model calls: 400; tokens: 155,085.
- Prior unrefined router+rewrite baseline: 400 calls; 149,398 tokens.
- The earlier partial-history attempt is in ignored `results-heavy-typos-controlled-refine-100/`; the corrected full-history rerun is in ignored `results-heavy-typos-controlled-refine-fullctx-v2-100/`. Raw row/detail outputs remain local.
- Dollar cost is omitted because exact billed model/account rates were not independently verified.
