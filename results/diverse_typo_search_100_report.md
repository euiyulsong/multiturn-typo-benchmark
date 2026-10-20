# Retrieval under more diverse synthetic typos

## What changed in the inputs?

`clean` is the original frozen dialogue and final user question, with **no new synthetic edits**. It can still contain natural spelling or grammar errors from the source dataset.

`typo` uses the exact same 100 cases and deterministically edits one eligible word in each user turn, including the target question. For each selected word, one of four character-level operations is chosen with equal probability:

- **Transpose:** swap two adjacent, different letters (`offer` → `offre`).
- **Delete:** remove one letter (`school` → `schoo`).
- **Insert:** insert a random lowercase letter (`obtain` → `obtawn`).
- **Replace:** replace one letter with a different letter (`obtain` → `obtawn`).

The operations are reproducible from case ID, turn position, and seed `20261020`. Existing agent turns, gold answers, gold documents, and gold spans are never changed. This is a synthetic stress set, not a claim that these operation frequencies match real user typing.

There were **408 edits across 100 typo cases**, or 4.08 edits per case on average. The realized counts were insert 114, delete 105, transpose 99, and replace 90. The counts are close to balanced; each operation is sampled independently for each eligible utterance.

## Setup

- Same frozen sample as the prior runs: 100 cases; 25 each from DMV, SSA, VA, and StudentAid.
- 100 clean runs + 100 diverse-typo runs = **100 complete matched pairs / 200 evaluations**.
- `gpt-6-luna`, direct router, `raw` (no separate correction), typo-aware router instruction on, direct/no-ReAct; retrieval-only, so no final answer generation.
- Hybrid retrieval scores: BM25, dense, RRF, cross-encoder; all rows use the same local index and rankers.
- Offline dense-index embedding batch size 25; up to 50 independent case evaluations in parallel.

## Results: diverse typo minus clean

Recall@5 is the share of gold documents/spans present in the top five ranked chunks. MRR@10 rewards gold documents appearing near the top ten. Delta is the mean paired per-case score difference, and W/T/L counts cases with higher/equal/lower scores in the typo condition.

| Metric | Clean mean | Diverse typo mean | Paired delta | W / T / L |
|---|---:|---:|---:|---:|
| Router tool-set EM | 0.930 | 0.930 | 0.000 | 1 / 98 / 1 |
| BM25 document Recall@5 | 0.700 | 0.660 | −0.040 | 4 / 88 / 8 |
| BM25 span Recall@5 | 0.558 | 0.503 | −0.055 | 3 / 88 / 9 |
| BM25 MRR@10 | 0.565 | 0.546 | −0.019 | 19 / 60 / 21 |
| Dense document Recall@5 | 0.570 | 0.590 | +0.020 | 7 / 88 / 5 |
| Dense span Recall@5 | 0.453 | 0.423 | −0.030 | 5 / 88 / 7 |
| Dense MRR@10 | 0.507 | 0.484 | −0.023 | 14 / 68 / 18 |
| RRF document Recall@5 | 0.660 | 0.680 | +0.020 | 8 / 86 / 6 |
| RRF span Recall@5 | 0.508 | 0.533 | +0.025 | 8 / 86 / 6 |
| RRF MRR@10 | 0.540 | 0.531 | −0.009 | 19 / 60 / 21 |
| Cross-encoder document Recall@5 | 0.720 | 0.720 | 0.000 | 4 / 92 / 4 |
| Cross-encoder span Recall@5 | 0.608 | 0.540 | −0.068 | 3 / 87 / 10 |
| Cross-encoder MRR@10 | 0.552 | 0.562 | +0.010 | 21 / 62 / 17 |

The clearest decrease is **BM25 Recall@5** (−0.04 documents and −0.055 spans); cross-encoder span Recall@5 is also lower (−0.068). Dense and RRF document recall are slightly higher, while their rank/span scores are mixed. Router EM stays the same. So the broader typo types make lexical retrieval and exact passage retrieval look more vulnerable, but the overall retrieval pipeline does not uniformly decline. Most per-case comparisons still tie on discrete Recall@5.

## Comparison with adjacent-transposition-only inputs

The earlier typo-aware direct-router run used the same 100 cases and configuration but only transposed adjacent letters. Here are selected deltas from clean:

| Metric | Transposition-only delta | Diverse-typo delta |
|---|---:|---:|
| BM25 document Recall@5 | +0.010 | **−0.040** |
| BM25 span Recall@5 | −0.020 | **−0.055** |
| Dense document Recall@5 | +0.050 | +0.020 |
| Dense span Recall@5 | +0.063 | −0.030 |
| RRF document Recall@5 | −0.010 | +0.020 |
| Cross-encoder document Recall@5 | +0.010 | 0.000 |
| Cross-encoder span Recall@5 | +0.010 | **−0.068** |
| Cross-encoder MRR@10 | −0.026 | +0.010 |

This supports the suspicion that the original transposition-only errors were relatively forgiving for lexical/document retrieval. Under the broader mix, BM25 and cross-encoder evidence-span recall decrease instead of staying flat or slightly increasing. The pattern is still not uniform across retrievers.

## Exact router prompts used

Both conditions used the same typo-aware direct-router system prompt. Domain descriptions are included in place of the runtime-generated tool list:

> You are a multi-tool search router. Pick one or more useful tools to answer the final user question. Use full dialogue for context. Select no more than two, deduplicate, and order by importance. Search tools: dmv = driver's license, registration, state motor vehicles; ssa = Social Security, retirement, benefits, disability; va = US Veterans Affairs and veterans benefits; studentaid = federal student aid, FAFSA, grants and loans. Write a focused standalone retrieval query resolving references only as needed. Return JSON with tools and query. Never use dataset labels. The input may contain spelling errors or typos: infer the intended words from dialogue context, and produce a corrected, focused standalone retrieval query without changing intent.

The model received the full dialogue (earlier turns plus final question) as user input. It was **not** given an explicit separate spelling-correction call in this run; it was instructed to infer/correct spelling while selecting the tool and writing the search query. Output was a strict JSON object with `tools` and `query` fields.

## Fairness and limitations

- Every case appears once in both clean and typo conditions; the retrieval pipeline, prompts, model, and gold labels are otherwise fixed. The only intended input difference is the synthetic edit operation mix.
- The operation types are equally likely per eligible word, but real-world typing errors are not necessarily equally distributed. This tests variety, not empirical error prevalence.
- Each eligible user turn gets one edit, so longer histories naturally receive more errors. Counts per example may vary with dialogue length.
- These are English, character-level errors only—not word splits/joins, homophones, OCR, speech recognition, grammar changes, or multilingual errors.
- The model is typo-aware, which may mitigate errors during query generation. Results therefore measure diverse input typos with an instructed typo-aware router, not an unaware search system.
- This report uses 100 cases from four domains and reports descriptive paired changes without confidence intervals. It tests search/routing only; no generated response was judged.

## Run record

- Evaluations: 200 (100 clean + 100 typo; 100 matched pairs)
- Model: `gpt-6-luna`
- Recorded LLM calls: 200; tokens: 84,306 (clean 41,703; typo 42,603)
- Average recorded model latency: 3.63 seconds/evaluation
- Typos: 408 total across typo cases (average 4.08/case)
- Embedding batch size: 25; maximum concurrent case evaluations: 50
- Frozen sample seed: `20261020`; run output directory `results-diverse-typos-100/` is local and ignored by Git.
- Dollar spend is not stated; exact billed model/account rates were not independently verified.
