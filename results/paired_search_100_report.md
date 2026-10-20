# Paired retrieval comparison: clean versus synthetic typos

## Result at a glance

We evaluated the **same frozen 100 dialogue cases** in both conditions: once with the source wording (`clean`) and once with synthetic adjacent-letter swaps (`typo`). The sample has 25 cases from each domain (DMV, SSA, VA, and StudentAid), giving **100 complete matched pairs / 200 evaluations**.

This is a retrieval-focused experiment. We stopped after the first retrieval and skipped answer generation and ReAct decisions. Both conditions used the same `gpt-6-luna` model, direct router, typo-aware router prompt, raw/no-separate-correction arm, and local hybrid retrieval pipeline. Therefore the contrast estimates the effect of synthetic typos **for this one configuration**; it does not compare correction strategies or agent architectures.

| First-search metric | Clean mean | Typo mean | Paired change (typo − clean) | Cases improved / tied / declined |
|---|---:|---:|---:|---:|
| Router tool-set EM | 0.930 | 0.930 | 0.000 | 1 / 98 / 1 |
| BM25 document Recall@5 | 0.670 | 0.680 | +0.010 | 6 / 89 / 5 |
| Dense document Recall@5 | 0.560 | 0.610 | +0.050 | 10 / 85 / 5 |
| RRF document Recall@5 | 0.650 | 0.640 | −0.010 | 5 / 89 / 6 |
| Cross-encoder document Recall@5 | 0.670 | 0.680 | +0.010 | 11 / 79 / 10 |
| Cross-encoder span Recall@5 | 0.528 | 0.538 | +0.010 | 9 / 83 / 8 |
| BM25 MRR@10 | 0.512 | 0.510 | −0.002 | 26 / 52 / 22 |
| Dense MRR@10 | 0.478 | 0.487 | +0.010 | 21 / 64 / 15 |
| RRF MRR@10 | 0.498 | 0.524 | +0.026 | 32 / 49 / 19 |
| Cross-encoder MRR@10 | 0.549 | 0.524 | −0.026 | 19 / 58 / 23 |

**Bottom line:** with a typo-aware router, adding synthetic transposition typos did not meaningfully change overall router accuracy or final top-five retrieval recall (cross-encoder +0.01). The dense retrieval and RRF rank metrics move in different directions from the cross-encoder MRR, so the pattern is mixed rather than a consistent typo penalty or benefit. Per-domain results vary: DMV cross-encoder Recall@5 falls by 0.08, SSA rises by 0.12, VA rises by 0.04, and StudentAid falls by 0.04. These domain slices each contain only 25 cases and are descriptive.

## What “retrieval only” means

For each case, the router reads the dialogue and chooses a search tool plus standalone query. The local system then runs BM25 and dense retrieval, fuses their rankings with RRF, reranks with a cross-encoder, and scores the initial result list. It does **not** ask the model to compose an answer. This removes one API call per evaluation and focuses evaluation on whether relevant documents/passages were found.

The main metrics:

- **Router tool-set EM:** exact match between chosen search-tool set and annotated gold tool set. In this dataset it mostly means selecting the right domain.
- **Document Recall@5:** fraction of annotated gold documents represented in the top five retrieved chunks. The cross-encoder value is after reranking.
- **Span Recall@5:** fraction of annotated gold support passages covered in the top five chunks. It is stricter than finding the right document.
- **MRR@10:** rewards ranking a gold document near the top; reciprocal rank is 1/rank of the first gold document, or zero if none appears in the top ten.
- **Paired change:** for each case, typo score minus clean score, then averaged over the same 100 cases. Positive means the typo input scored higher on that metric; negative means lower. “Improved/tied/declined” counts refer to the sign of that per-case difference.

These metrics do not assess response correctness, grounding quality of a generated answer, or whether correcting typos would help. They only assess routing and retrieval under the selected prompt/configuration.

## Fairness and caveats

- Every one of the 100 frozen cases is present in both groups exactly once, and all fixed pipeline factors are the same. This is a balanced, paired comparison of clean vs. typo **within the chosen configuration**.
- It is not a randomized model-output repeat. Each condition is evaluated once per case; generation variability is not measured, though answer generation is omitted here.
- The `typo` condition applies deterministic adjacent-character transpositions to eligible words in user turns. There are on average **4.08 edits per case**; `clean` means no new synthetic edits and may still contain natural errors from the source dialogue.
- The typo-aware prompt is enabled for both arms. This answers how typo presence affects search when the router is explicitly told to tolerate typos. It does not compare typo-awareness on vs. off.
- The sample is the benchmark's frozen validation sample, not a random draw from all possible support conversations. The results generalize only cautiously beyond these four domains and this corpus.
- Recall@5 and MRR differ in what they reward; no single retrieval number captures every kind of search quality. The mixed metric pattern should be reviewed alongside per-case examples before changing the system.
- No confidence interval or significance claim is made. Paired case changes are shown to make the small deltas and large number of ties visible; formal uncertainty can be added if needed.

## Run and cost record

- Model: `gpt-6-luna`
- Frozen cases: 100 (25/domain); conditions: clean and typo; **200/200 evaluations complete**
- Configuration: `raw`, `direct` architecture, direct/no-ReAct, typo-aware prompt on, first retrieval only
- Recorded model calls in saved rows: **214** (some first-attempt rows were replaced by retrieval-only reruns; cached responses were reused without a new call)
- Reported model tokens in final saved rows: **110,016**
- Sum of recorded API-call latency in final saved rows: **763.6 seconds**; local embedding/reranking setup and runtime are additional
- Output files are kept in local `results-paired-100/`, excluded from Git because details contain dialogue text.
- Reproducibility: seed `20261020`; fixed case IDs are the 100 IDs in `results/case_ids.txt`.

Dollar cost is not stated because the exact billed rate/discount for this model and account was not verified. The token total is reported for transparency.
