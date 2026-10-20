# Retrieval noise without a router: clean vs. diverse typos

## What this comparison isolates

This is a **retriever-only diagnostic**, not an end-to-end agent run. We skipped the LLM router entirely and used each case's annotated gold domain to select the domain-specific index. The retriever received the complete raw dialogue text (earlier turns plus final question). Thus, this comparison measures sensitivity of BM25/dense/reranking to input spelling noise **when routing is held correct and no LLM rewrites or corrects the query**.

The same 100 frozen cases (25 each from DMV, SSA, VA, and StudentAid) were searched in both conditions: 100 clean runs plus 100 diverse-typo runs = **100 complete matched pairs / 200 searches**. The typo edits are the same deterministic transpose/delete/insert/replace variants as [`diverse_typo_search_100_report.md`](diverse_typo_search_100_report.md): 408 edits total, 4.08 per case on average.

## Results

Recall@5 is the fraction of gold documents/passages present in the top five. MRR@10 rewards ranking a gold document near the top ten. Delta is the mean per-case typo score minus clean score; W/T/L counts cases where the typo score was higher/equal/lower.

| Metric | Clean | Diverse typo | Paired delta | W / T / L |
|---|---:|---:|---:|---:|
| BM25 document Recall@5 | 0.540 | 0.550 | +0.010 | 4 / 93 / 3 |
| BM25 span Recall@5 | 0.420 | 0.413 | −0.007 | 2 / 95 / 3 |
| BM25 MRR@10 | 0.399 | 0.354 | **−0.045** | 12 / 66 / 22 |
| Dense document Recall@5 | 0.600 | 0.570 | −0.030 | 2 / 93 / 5 |
| Dense span Recall@5 | 0.413 | 0.393 | −0.020 | 2 / 94 / 4 |
| Dense MRR@10 | 0.475 | 0.452 | **−0.023** | 11 / 74 / 15 |
| RRF document Recall@5 | 0.640 | 0.640 | 0.000 | 4 / 92 / 4 |
| RRF span Recall@5 | 0.530 | 0.488 | −0.042 | 4 / 87 / 9 |
| RRF MRR@10 | 0.518 | 0.489 | −0.029 | 11 / 72 / 17 |
| Cross-encoder document Recall@5 | 0.620 | 0.650 | +0.030 | 5 / 93 / 2 |
| Cross-encoder span Recall@5 | 0.468 | 0.443 | −0.025 | 3 / 92 / 5 |
| Cross-encoder MRR@10 | 0.509 | 0.492 | −0.018 | 14 / 67 / 19 |

The strongest consistent warning sign is **ranking quality**: BM25 MRR@10 falls by 0.045 and dense MRR@10 by 0.023. Dense document and span Recall@5 also fall by 0.030 and 0.020. BM25 document Recall@5 is nearly unchanged, and cross-encoder document Recall@5 is slightly higher, but cross-encoder span Recall@5 and MRR are lower. In short, a gold document often remains somewhere in the top five, but typos can make the best matching evidence less prominent or make exact support passages harder to retrieve.

Discrete Recall@5 has many ties (typically 92–95 of 100 cases). The MRR changes capture rank shifts even when the same gold document remains in the top ten.

## Comparison with the router-based run

The previous diverse-typo experiment used an LLM router that saw the full dialogue, selected a tool, and generated a focused typo-aware search query. In the router run, typos could be corrected or normalized before retrieval. This no-router run fixes the domain to gold and searches the **raw dialogue directly**. It removes router errors and query rewriting, so it answers a narrower question: “How noisy is retrieval when fed the uncorrected multi-turn text?”

The no-router clean baseline is lower on several metrics than the router-based clean baseline (for example BM25 document Recall@5 is 0.54 vs. 0.70). Do not interpret that difference as the router's causal gain: raw full-dialogue query length/content differs from a focused standalone router query, and the oracle domain is fixed. Compare typo vs. clean *within this no-router setup* for the retrieval-noise effect.

## Exact prompt and query setup

**No LLM prompt was used.** The router, query-rewrite stage, separate typo corrector, final answer generator, and ReAct steps were all skipped. For each case, the index was fixed to the gold domain from the dataset, and the search string was the original full dialogue concatenated in turn order as `ROLE: text`, including the target question. The typo condition changed eligible user-turn words only; agent turns and gold annotations were unchanged.

The same local BM25 (`rank_bm25`), dense (`sentence-transformers/multi-qa-MiniLM-L6-cos-v1`), RRF, and cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`) ranking pipeline was applied to both versions.

## Fairness and limitations

- Same 100 case IDs, each in clean and diverse-typo form; only synthetic edits differ within each pair.
- Gold-domain selection is an **oracle diagnostic**. Real deployment must infer the domain; this test deliberately removes routing as a confounder.
- The search query is the full raw dialogue, not a standalone rewritten question. Results are not directly comparable to router-based recall scores as if only routing were changed.
- The stress set chooses transpose/delete/insert/replace equally per eligible word; real typo frequencies may differ. There are on average 4.08 edits per case.
- English character-level misspellings only; no word splitting/joining, homophones, OCR, ASR, grammar changes, or multilingual noise.
- Descriptive paired means and W/T/L counts only; no confidence intervals or statistical-significance claim. The study measures retrieval, not answer quality.

## Run record

- 200 searches (100 clean + 100 typo), 100 matched pairs
- LLM calls/tokens: **0 / 0**; router skipped
- Gold domain fixed per case; batch size 25 for local dense index, concurrency setting 50 (unused because there are no LLM calls)
- Frozen sample seed `20261020`; outputs local in ignored `results-no-router-diverse-100/`
