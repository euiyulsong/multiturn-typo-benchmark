# Run status — 2026-10-20

## Intended main experiment

- Dataset: frozen 100-case MultiDoc2Dial validation sample, 25 dialogues per domain.
- Model requested: `gpt-6-luna`.
- Main factors: clean/typo input × typo-aware prompt off/on × staged/direct query+tool architecture × no-ReAct/ReAct; `raw` correction arm only.
- Target: 1,600 evaluations.
- Script: `python3 benchmark.py --out ./results --architecture both --typo-awareness both --agent both --fresh-run`

## Execution outcome

The API returned `429 insufficient_quota` (`credit_balance_exhausted`) before the current main run could write any scored rows. The 100-case sample and experiment code are prepared, but **the 1,600-evaluation main experiment is incomplete (0/1,600)**. No conclusions should be drawn from the mixed rows in `rows.jsonl`.

`rows.jsonl` and `details.jsonl` include 771 legacy results from the earlier protocol and a partial 202-row attempt from an incorrect factorial setup. They do not have a consistent current-protocol coverage and must not be presented as final results. They are excluded from public version control. The fresh-run mode uses separate output filenames and summary directory to avoid mixing them.

## Evaluation metric correction

- `answer_em`: normalized exact match to free-form gold agent response; strict and often zero for valid paraphrases.
- `answer_token_f1`: lexical overlap against gold agent response.
- `answer_subspan_em`: best token-F1 against annotated source support span(s), with gold-answer exact-match fallback.
- `final_subspan_em@5`: fraction of annotated source spans exactly present in top-5 retrieved chunks.

## Legacy exploratory analysis

See [`legacy_analysis.md`](legacy_analysis.md) for an explicitly exploratory summary of the local 973-row legacy snapshot. It defines the metrics in plain language and audits comparison fairness: the rows are mixed, unbalanced, partly duplicated, and incompletely paired, so they cannot fairly establish a winning strategy. Predictions were scored with **best-subspan EM** (maximum normalized token F1 against an annotated supporting span) because strict response EM was 0/973. The legacy snapshot is not evidence for the incomplete current main experiment; the raw JSONL files and source archive remain excluded from Git.

Offline mock tests validate code paths only and are not measured benchmark results.
