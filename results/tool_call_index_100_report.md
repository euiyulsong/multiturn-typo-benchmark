# Tool-call routing comparison: clean vs. diverse typos

## Scope

This report evaluates the **initial tool-selection call** on the same 100 frozen dialogue cases, once with source input (`clean`) and once with diverse synthetic typos (`typo`): 100 complete matched pairs / 200 router outputs. It reuses the router outputs from [`diverse_typo_search_100_report.md`](diverse_typo_search_100_report.md), so **no additional model calls** were made.

The dataset provides a gold domain-specific search tool for each turn, usually one tool. It does not provide a reference tool-call argument/query string, so this measures tool-set routing, not exact search-query correctness or general multi-tool API argument accuracy.

## Prompt and inputs

The router saw the complete dialogue history plus the final question. The typo condition includes one character-level edit per eligible user utterance, with the edit chosen among transpose, delete, insert, and replace. Agent turns and gold labels remain unchanged.

The typo-aware direct-router system prompt was the same in both conditions:

> You are a multi-tool search router. Pick one or more useful tools to answer the final user question. Use full dialogue for context. Select no more than two, deduplicate, and order by importance. Search tools: dmv = driver's license, registration, state motor vehicles; ssa = Social Security, retirement, benefits, disability; va = US Veterans Affairs and veterans benefits; studentaid = federal student aid, FAFSA, grants and loans. Write a focused standalone retrieval query resolving references only as needed. Return JSON with tools and query. Never use dataset labels. The input may contain spelling errors or typos: infer the intended words from dialogue context, and produce a corrected, focused standalone retrieval query without changing intent.

## Results

| Metric | Clean | Diverse typo | Paired change (typo − clean) |
|---|---:|---:|---:|
| Tool-set exact match (EM) | 93/100 (0.93) | 93/100 (0.93) | 0 cases net |
| Tool-set F1 | 0.957 | 0.957 | 0.000 |
| Number of selected tools = 1 | 93 | 94 | +1 |
| Number of selected tools = 2 | 4 | 4 | 0 |
| Number of selected tools = 0 (invalid/empty raw output) | 3 | 2 | −1 |

Across the 100 matched cases, the tool-set exact match changed on **2 cases**: one moved from incorrect to correct, and one from correct to incorrect. The other 98 cases retained the same exact-match status. Tool F1 increased on one case, decreased on one, and tied on 98.

## Interpretation

For this fixed sample, model, and typo-aware prompt, the broader synthetic typo mix did **not meaningfully change tool routing**: exact match and F1 are identical in aggregate. This does not mean every route was right—7 clean and 7 typo outputs failed exact match. It means typo edits rarely changed whether the selected domain-tool set matched the dataset label.

Search quality can still shift after routing. BM25/dense/RRF/cross-encoder retrieval metrics in the companion report evaluate ranked evidence after the router returns tools and a query; routing EM alone does not measure those stages.

## Fairness and limitations

- Same fixed 100 dialogue cases, each evaluated once per input condition; clean/typo results are paired case-by-case.
- Same model (`gpt-6-luna`), router prompt, domain tools, decoding/schema, and evaluation labels in both conditions.
- The model was explicitly told to infer typos. This tests typo robustness **with typo-aware instructions**, not an unaware baseline.
- Synthetic errors are equally selected among character transpose/delete/insert/replace; they do not represent measured real-world typo frequencies.
- Most cases have a single gold tool. This is domain routing, not a benchmark of complex multi-tool composition or exact API argument strings.
- One model output per case and no confidence interval; treat this as descriptive evidence, not a statistical equivalence test.

## Run record

- 200 router outputs (100 clean + 100 typo); no answer generation or ReAct calls.
- Reused from local run `results-diverse-typos-100/`; no additional API calls.
- Model: `gpt-6-luna`; 25 cases per domain; frozen sample seed `20261020`.
- The raw details and dialogue text remain local and ignored by Git; this report includes aggregate routing metrics only.
