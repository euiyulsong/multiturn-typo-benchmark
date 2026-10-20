# Retrieval comparison: router typo handling vs. explicit typo correction

## What we ran

We reused the same **100 frozen cases** (25 each from DMV, SSA, VA, and StudentAid), with each case evaluated in clean and synthetic-typo form. That gives 100 matched clean/typo pairs for each setup.

1. **Router-only typo handling (`raw` + `direct`):** the direct router sees the whole dialogue, is explicitly told it may contain typos, and writes the search query/tool in one call. No separate correction call. Results are from [`paired_search_100_report.md`](paired_search_100_report.md).
2. **Explicit last-turn correction (`last` + `staged`):** a dedicated correction call first fixes only the last user question; then the router (with the same typo-aware instruction) chooses the search query/tool. Search is retrieval-only; answer generation and ReAct are skipped.

This is a same-case comparison across two pipeline configurations. “Clean vs typo” within each setup is paired. “Explicit correction vs router-only” is also case-matched on the 100 typo inputs, but changes the correction stage and routing architecture together, so it is a practical pipeline comparison—not an isolated causal estimate of spelling correction alone.

## Did typos hurt retrieval?

The table shows **typo minus clean** differences within each setup. Recall@5 is the share of gold documents/passages present in the top five; MRR@10 rewards a gold document appearing near the top ten. Positive differences mean the typo condition scored higher. W/T/L is the count of cases that improved/tied/declined.

### No separate correction: typo-aware direct router

| Metric | Clean | Typo | Paired delta | W / T / L |
|---|---:|---:|---:|---:|
| Router tool-set EM | 0.930 | 0.930 | 0.000 | 1 / 98 / 1 |
| BM25 document Recall@5 | 0.670 | 0.680 | +0.010 | 6 / 89 / 5 |
| BM25 span Recall@5 | 0.543 | 0.523 | −0.020 | 5 / 88 / 7 |
| BM25 MRR@10 | 0.512 | 0.510 | −0.002 | 26 / 52 / 22 |
| Dense document Recall@5 | 0.560 | 0.610 | +0.050 | 10 / 85 / 5 |
| Dense span Recall@5 | 0.400 | 0.463 | +0.063 | 10 / 87 / 3 |
| Dense MRR@10 | 0.478 | 0.487 | +0.010 | 21 / 64 / 15 |
| Cross-encoder document Recall@5 | 0.670 | 0.680 | +0.010 | 11 / 79 / 10 |
| Cross-encoder span Recall@5 | 0.528 | 0.538 | +0.010 | 9 / 83 / 8 |
| Cross-encoder MRR@10 | 0.549 | 0.524 | −0.026 | 19 / 58 / 23 |

### Explicitly correct the last turn, then route/search

| Metric | Clean | Typo | Paired delta | W / T / L |
|---|---:|---:|---:|---:|
| Router tool-set EM | 0.910 | 0.910 | 0.000 | 2 / 96 / 2 |
| BM25 document Recall@5 | 0.660 | 0.680 | +0.020 | 6 / 90 / 4 |
| BM25 span Recall@5 | 0.543 | 0.550 | +0.007 | 5 / 91 / 4 |
| BM25 MRR@10 | 0.515 | 0.541 | +0.026 | 16 / 76 / 8 |
| Dense document Recall@5 | 0.550 | 0.510 | −0.040 | 2 / 92 / 6 |
| Dense span Recall@5 | 0.453 | 0.443 | −0.010 | 4 / 91 / 5 |
| Dense MRR@10 | 0.439 | 0.466 | +0.027 | 12 / 80 / 8 |
| Cross-encoder document Recall@5 | 0.710 | 0.740 | +0.030 | 7 / 89 / 4 |
| Cross-encoder span Recall@5 | 0.630 | 0.663 | +0.033 | 5 / 92 / 3 |
| Cross-encoder MRR@10 | 0.593 | 0.604 | +0.011 | 8 / 79 / 13 |

## Does explicit correction help on typo inputs?

On the same 100 typo cases, compare `last` correction against the direct typo-aware router:

| Search metric | Router-only direct | Explicit last correction | Paired change with correction |
|---|---:|---:|---:|
| BM25 document Recall@5 | 0.680 | 0.680 | 0.000 |
| BM25 span Recall@5 | 0.523 | 0.550 | +0.027 |
| BM25 MRR@10 | 0.510 | 0.541 | +0.032 |
| Dense document Recall@5 | 0.610 | 0.510 | −0.100 |
| Dense span Recall@5 | 0.463 | 0.443 | −0.020 |
| Dense MRR@10 | 0.487 | 0.466 | −0.021 |
| RRF document Recall@5 | 0.640 | 0.660 | +0.020 |
| Cross-encoder document Recall@5 | 0.680 | 0.740 | +0.060 |
| Cross-encoder span Recall@5 | 0.538 | 0.663 | +0.125 |
| Cross-encoder MRR@10 | 0.524 | 0.604 | +0.081 |

The strongest apparent gain is **after cross-encoder reranking**: last-turn correction increases typo-case span Recall@5 by 0.125 and MRR@10 by 0.081 on average. However, first-stage dense Recall@5 falls by 0.10; BM25 document Recall@5 is unchanged. This suggests the corrected pipeline changes which candidates are retrieved/ranked, and the reranker recovers more gold evidence—it does **not** mean every retriever improves. Router EM is 0.02 lower with this setup (0.91 vs. 0.93), despite the reranked retrieval gains.

## Interpretation

- The “typo is too easy” suspicion is partly consistent with the first experiment: **with typo-aware routing and no separate correction, overall document Recall@5 changes very little** (BM25 +0.01; cross-encoder +0.01). Many per-case values tie exactly.
- Explicit correction provides a more useful contrast. With the same typo-bearing cases, last-turn correction plus staged routing gives higher cross-encoder evidence retrieval, especially span Recall@5, than the direct router-only setup. It does not improve every component: dense top-five document recall is lower and router accuracy is slightly lower.
- Within the explicit-correction condition, typo vs. clean remains mostly stable on Recall@5; cross-encoder document and span recall rise by +0.03. This does not establish that typos are beneficial—per-case differences are mostly ties and sampling/model variability remains.
- The observed BM25/dense/cross-encoder disagreement is informative: typo correction may affect exact token matching (BM25), semantic search (dense), fusion/reranking, and routing differently. Look at query and case-level outputs before treating a single aggregate as the system result.
- No answer was generated, so none of these findings establish response quality, answer grounding, or user-facing correctness.

## Fairness, caveats, and costs

- Both pipeline setups use exactly the same 100 case IDs and the same clean/typo construction, retrieval index/models, typo-aware router instruction, and model ID. The 200 evaluations within each setup form 100 complete clean/typo pairs.
- The explicit-correction comparison changes two factors (`direct` to `staged`, and adds a correction call) by design. The result answers “does this correction-plus-routing pipeline help?” rather than “what is the pure correction-only effect with every other model input held identical?”
- Correction only targets the **last user turn**. It leaves typos in earlier history untouched. `clean` means no newly injected synthetic typo and may contain natural source-language errors.
- The typo-aware instruction remains on in both setups. This measures an extra correction stage on top of a typo-aware router; it does not compare typo-aware prompt on vs. off.
- Synthetic errors are adjacent-character transpositions only, averaging 4.08 edits in each typo case. Results do not cover OCR, ASR, phonetic, or other typo types.
- The sample is 100 frozen validation cases from four domains. W/T/L counts and means are descriptive; there is no confidence interval or significance claim here.
- The explicit-correction run used local embedding `--batch-size 25` and up to **25 concurrent case evaluations**. Its saved rows record 600 API calls and 224,430 tokens; the no-separate-correction run records 214 calls and 110,016 tokens. Concurrency shortens wall time, not token usage or billed cost. Dollar spend is omitted because the exact model/account rates were not verified.
- Row-level JSONL and detailed dialogue text remain local in ignored directories. Reports include aggregate metrics only.
