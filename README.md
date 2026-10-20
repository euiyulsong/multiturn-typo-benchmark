# Multi-turn typo correction × multi-tool router × hybrid retrieval × max-3-step ReAct

**Status: frozen 100-case sample; full benchmark blocked by exhausted model API credits.**

Preparation was run on 2026-10-20 with seed `20261020`: 100 cases selected
(25 per domain: `dmv`, `ssa`, `va`, `studentaid`). Offline synthetic harness
checks passed. The gpt-6-luna API returned `429 insufficient_quota` before the
fresh main evaluation produced any rows. A previous legacy run contains 973
rows (771 with no architecture/prompt-factor metadata, plus partial rows from
the old factorial attempt); these rows are not a complete or valid result for
the current design. Do not use them to draw conclusions. The final script now
supports `--fresh-run` to prevent mixing previous rows into future outputs.

## Design
- Data: 100 distinct MultiDoc2Dial validation dialogues, 25 from each of `dmv`, `ssa`, `va`, `studentaid`, chosen with deterministic SHA256 ordering. For each dialogue, choose one user question that follows a prior user question and has an annotated agent response with grounding spans.
- Two **paired inputs** per case: unmodified source conversation (`clean`; natural typos in the source remain) and synthetic typo injection (`typo`; exactly one adjacent-character transposition on each eligible user utterance, with unchanged agent turns). Corpus and labels are never corrupted.
- Correction arms: `raw` (no separate correction), `last` (LLM spell-corrects only the last user utterance), `full` (LLM corrects all user utterances). `--strategy raw` runs the primary hypothesis design; default `--strategy all` runs each correction arm. In the direct architecture the correction arms are intentionally bypassed, so these are redundant controls there.
- Factorial extension: `typo_aware=false/true` adds/removes an explicit typo-interpretation instruction. The `staged` architecture applies the selected correction arm, then makes a separate query-rewrite call and a separate tool-selection call. The `direct` architecture bypasses those preprocessing/rewrite calls and makes one structured call jointly select tools and produce a typo-robust standalone query. For a clean ablation of the prompt instruction, compare `strategy=raw` with typo-aware off vs on; `last/full` estimate whether an extra correction call still adds value.
- Full design including optional correction ablations is 100 paired cases × clean/typo × raw/last/full × ReAct on/off × staged/direct architecture × typo-aware off/on = **4,800 evaluations**. The primary hypothesis design uses the `raw` arm only and is **1,600 evaluations**. ReAct includes the initial search and up to two additional tool searches (three total); direct means no ReAct loop and is retained as a separate baseline. If a run is interrupted, `--fresh-run` saves independently scoped fresh rows and summary; legacy rows are not mixed into that output.
- Initial multi-tool router selects up to 2 domain-specific retrieval functions and one query; all arms use the **same** four tools and indexes. Multi-tool **selection** is evaluated, but most ground truth cases have one gold domain/tool, so this dataset does not establish performance on inherently multi-tool tasks.
- Search: per-domain **BM25** top-30 + `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` dense top-30; **RRF** k=60 top-20; `cross-encoder/ms-marco-MiniLM-L6-v2` reranks; evidence top-5 to LLM. Fixed chunking 1,050 characters / 180 overlap, same document corpus across all arms. No gold/reference info is placed in model prompts or retrieval.
- `direct`: first retrieval → final answer. `react`: first retrieval → inspect evidence → optional new query/tool → up to **three search steps total** → final answer. First retrieval is identical for direct and ReAct for a given correction arm. React additional search decisions cost LLM calls.
- **Metrics:** initial exact tool-set EM and tool F1; document Recall@1/3/5/10, MRR@10 and span coverage by BM25/Dense/RRF/Cross-Encoder; final document/span Recall@1/3/5 plus retrieval `final_subspan_em@5` (fraction of annotated source spans present in the top-five evidence); **offline step checkpoint Answer EM/F1 after each observed ReAct search step**; next-agent-response normalized EM, answer subspan EM (best token F1 against annotated supporting spans), and token F1; cited evidence-doc set EM; LLM calls/tokens; wall-clock phase latency and p95. Paired bootstrap (2,000 draws) of per-case deltas to raw. Logged JSONL includes per-step search results and correction output.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='...'
export BENCH_MODEL='your-API-accessible-model-id'   # confirm model access; not a ChatGPT UI label
python benchmark.py --out ./results --device cpu
```

The program downloads the official dataset from https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip. If unavailable, download it yourself and use `--data-zip /path/to/multidoc2dial.zip`.

Inspect selection before invoking the API:

```bash
python benchmark.py --out ./results --prepare-only
```

The selected sample and configuration from the preparation run are tracked at
[`results/selected_cases.json`](results/selected_cases.json) and
[`results/run_config.json`](results/run_config.json). The downloaded source
archive under `results/data/` is intentionally ignored; the script downloads
it again when needed, or you can pass `--data-zip`.

Smoke test 2 scenarios first (one dialogue × 12 conditions, or domain sampling), then remove the limit:

```bash
python3 benchmark.py --out ./smoke --limit 1
```

Run the offline structural checks with `python3 test_harness.py`. The harness
uses synthetic data and a mock model, so its output must never be presented as
measured benchmark performance.

A complete run writes `results/selected_cases.json`, `results/run_config.json`, `results/rows.csv`, `results/summary.csv`, `results/paired_deltas.csv`, `results/step_summary.csv`, `results/report.md`, `results/rows.jsonl`, `results/details.jsonl`, and `results/model_cache.jsonl`. Output files are written **only if the experiment was executed**. Never treat the template as a completed result.

## Important limitations / fairness notes
1. MultiDoc2Dial has **free-form agent replies**, so response EM will usually be harsh/low even for correct grounded paraphrases. Use token F1, cited doc EM and retrieval span coverage alongside EM. Corpus-grounded response evaluation is *not* a factoid-answer dataset.
2. The four tool choices are domain-specific search. The dataset usually requires one domain per agent turn, so `router_em` is primarily domain routing, **not** BFCL-level multi-function argument accuracy. For true multi-function composition, use a separate BFCL-V4 evaluation with compatible tool environments and do not blend its scores into these 100 cases.
3. This is a restricted four-domain document corpus, not web-scale QA. Retrieval scores do not transfer directly to production.
4. Synthetic typo types (letter transpositions only) do not cover ASR, OCR, phonetic or Korean errors. Add stress sets separately for those and mark them distinctly.
5. Full-history correction changes multiple user utterances at once. Do not conflate spelling correction with query rewrite/coreference resolution; the router alone creates the standalone retrieval query in every arm.
6. Each correction/route/decision/answer request uses the same chosen Responses API model. Model IDs and entitlements differ; the user must supply a valid `BENCH_MODEL` to run. The current full run was blocked by API credit exhaustion; see `results/RUN_STATUS.md`.
7. Evaluation arm order is shuffled with a fixed seed and query embeddings are not cached, reducing order-driven latency bias. Model output caching persists for resumed runs and reuses original measured API duration.
8. Intermediate checkpoint answers are offline probes: they are never fed into ReAct and are excluded from its production runtime, calls, and token usage. Their hypothetical EM is recorded in `details.jsonl` and per-case rows; absence of step 2/3 means the agent stopped earlier.
9. Cached API *latency is the recorded original* response latency, even on replays; retrieval timing is local execution time. For rigorous deployment p95, run several repeats on target hardware with cache disabled, controlling warmup and concurrency.
10. Token usage/cost and caching matter; do not choose a production arm based on EM alone. Production choice should be made on typo-specific paired deltas and end-to-end latency, with confidence intervals.

## Experiment status (2026-10-20)

The frozen sample contains 100 MultiDoc2Dial validation dialogues (25 each in DMV, SSA, VA, and StudentAid); all have multi-turn history. Their IDs are recorded in `results/case_ids.txt`; `results/experiment_manifest.json` records seed, run factors, model and status without publishing source dialogue text. The harness ran against gpt-6-luna, but the account has no remaining API credits and returned `429 insufficient_quota`, leaving the **current main run with 0/1,600 completed evaluations**. Existing local row-level logs are an older mixed partial run and are excluded from version control. The extended harness compiles and `python3 test_harness.py` passes synthetic structural checks; those checks are not model scores. See [`results/RUN_STATUS.md`](results/RUN_STATUS.md) for the run record.

An exploratory write-up of the local legacy 973-row snapshot, including **best-subspan EM**, plain-language metric definitions, and a fairness audit, is in [`results/legacy_analysis.md`](results/legacy_analysis.md). The mixed, unbalanced, partly duplicated rows are not current benchmark results and must not be used to pick a winning strategy. Raw rows, details, and downloaded source archive remain local and are not published.

A balanced retrieval-only comparison of clean vs. synthetic typos over all 100 frozen cases (200 evaluations) is summarized in [`results/paired_search_100_report.md`](results/paired_search_100_report.md). Answer generation and ReAct were skipped to reduce calls. The initial 8-case smoke pilot is retained in [`results/paired_pilot_report.md`](results/paired_pilot_report.md); both sets of raw outputs remain local and ignored by Git.

The follow-up retrieval comparison with explicit last-turn typo correction (also over the same 100 cases, with clean/typo pairs) and BM25/dense/cross-encoder breakdown is in [`results/correction_search_100_report.md`](results/correction_search_100_report.md). Raw outputs remain local and ignored by Git.

An additional 100-pair retrieval-only stress run uses four balanced synthetic typo operations (transpose, delete, insert, replace) to test whether the earlier transposition-only inputs were too forgiving. Results, exact condition definition, and router prompt are in [`results/diverse_typo_search_100_report.md`](results/diverse_typo_search_100_report.md). Its row-level outputs are local and ignored by Git.

The initial tool-call routing comparison on those same diverse-typo outputs (clean vs. typo, exact tool-set match and F1) is summarized in [`results/tool_call_index_100_report.md`](results/tool_call_index_100_report.md). No extra model calls were needed.

To isolate retrieval noise from routing, [`results/no_router_search_noise_100_report.md`](results/no_router_search_noise_100_report.md) reports a gold-domain diagnostic that skips the LLM router and searches the raw dialogue directly, comparing clean and diverse typo inputs over the same 100 cases. No LLM calls were used in this diagnostic; routing fields are recorded as not applicable.

A stronger-noise variant applies three character edits to each selected user-turn word and repeats the same router-free gold-domain retrieval comparison; see [`results/heavy_typo_search_noise_100_report.md`](results/heavy_typo_search_noise_100_report.md). It uses the same 100 cases and also makes no LLM calls.

The follow-up heavy-typo run with router tool selection and query rewriting before retrieval (same 100 cases, batch/concurrency 100) is documented in [`results/heavy_typo_router_rewrite_100_report.md`](results/heavy_typo_router_rewrite_100_report.md).

The one-sentence-only refined-query variant (router selects tools, but retrieval uses only the refiner's sentence) is compared with the standard router-query path in [`results/heavy_typo_one_sentence_rewrite_100_report.md`](results/heavy_typo_one_sentence_rewrite_100_report.md), using the same heavy-typo cases and batch/concurrency 100.

The controlled refinement ablation compares rewriting the final question without history vs. giving the rewriter full dialogue context; both then use the same router-selected-tool/router-query retrieval path. Its paired results are in [`results/heavy_typo_controlled_refinement_100_report.md`](results/heavy_typo_controlled_refinement_100_report.md).

### What the completed retrieval experiments suggest

Across these experiments, a consistent practical conclusion is that **query formation matters at least as much as the character-level retriever**. The 100 fixed cases were paired between clean and typo inputs, and the stronger typo stress set applied three mixed character edits to one word in every eligible user turn. Results below are retrieval-only, use one `gpt-6-luna` output per condition/case, and are descriptive rather than statistically conclusive.

| Setup | Clean → typo summary | Takeaway |
|---|---|---|
| Typo-aware direct router, one transposition per eligible user turn | Router EM 0.93 → 0.93; cross-encoder doc Recall@5 0.67 → 0.68 | Very small aggregate change under this typo-aware setup. |
| Typo-aware direct router, mixed transpose/delete/insert/replace | Router EM 0.93 → 0.93; BM25 doc Recall@5 0.70 → 0.66; BM25 span Recall@5 0.558 → 0.503; cross-encoder span Recall@5 0.608 → 0.540 | Broader typo types expose more lexical and evidence-span sensitivity, though document-level retrieval remains mixed across rankers. |
| No router, raw full dialogue sent to search in the gold domain | BM25 MRR@10 0.399 → 0.354; dense MRR@10 0.475 → 0.452; RRF span Recall@5 0.530 → 0.488 | Without query rewriting/correction, heavy typos lower ranking and span coverage in several retrievers. This is an oracle-domain diagnostic, not end-to-end routing. |
| Router + typo-aware query rewrite, same heavy typo stress | Router EM 0.92 → 0.94; BM25 MRR@10 0.514 → 0.544; RRF MRR@10 0.519 → 0.569; cross-encoder span Recall@5 0.647 → 0.667 | The rewrite/router pipeline is robust on this sample and appears to buffer raw-input noise; slight dense MRR decline remains. |

Additional observations:

- **Routing was stable:** with the typo-aware prompt on, tool-set exact match was 0.93 on clean and typo for the diverse stress run; only 2 of the 100 paired cases changed exact-match status. This measures domain-tool selection, not query quality.
- **Explicit correction is mixed by retrieval stage:** correcting only the last user turn before the staged router raised cross-encoder typo-case span Recall@5 from 0.538 to 0.663 and MRR@10 from 0.524 to 0.604 versus the direct typo-aware pipeline. BM25 document Recall@5 stayed 0.68, while dense document Recall@5 fell from 0.61 to 0.51. The comparison changes both the correction stage and routing architecture, so it does not isolate correction alone.
- **The oracle-domain no-router test must not be read as the system score.** It uses the gold domain and raw full dialogue text, while router runs select tools and create a standalone query. Its role is to diagnose retriever sensitivity when routing is held out of the equation.
- **Do not conclude that typos improve retrieval** from individual positive cells. The metric pattern is mixed, discrete Recall@5 has many ties, and no confidence intervals/significance tests were computed. These are stress-test observations on one frozen sample, not estimates of real-world typo prevalence.
- None of these retrieval-only experiments measures answer correctness or grounding. The original planned 1,600-evaluation factorial remains unrun; correction, typo-awareness, architecture, and ReAct should be compared in that complete paired design before making a production choice.

Full methods, exact prompts, typo definitions, and detailed metric tables are linked above in the individual reports.

To run the primary 1,600-evaluation design, install dependencies and provide API access as above. Run raw-only, ReAct/no-ReAct, both architectures, and typo-awareness off/on:

```bash
python3 benchmark.py --out ./results --architecture both --typo-awareness both --agent both --fresh-run --strategy raw
```

This raw-only factorial is exactly 1,600 evaluations: 100 cases × 2 input types × 2 architectures × 2 typo instructions × 2 agent modes.

To run the extended correction ablations too (4,800 evaluations), omit `--strategy raw`.

```bash
python3 benchmark.py --out ./results --architecture both --typo-awareness both --agent both --fresh-run
```

This compares typo-aware prompt off/on, typo-free/typo input, staged/direct architecture, and ReAct/no-ReAct. Add API credits before running; it resumes/skips completed rows in fresh-run files.

The first query/selection call is shared between no-ReAct and ReAct within every condition; ReAct can then make up to two additional searches. The primary hypothesis test is the paired typo-only effect `typo_aware=true` vs `false` for `strategy=raw`, using router tool-set EM/F1, CE/final document and span recall, answer token F1, and latency/calls. Compare typo-aware raw with staged `last`/`full` to determine whether explicit typo awareness removes the incremental value of separate query correction. Report clean-input deltas as a regression check and paired bootstrap CIs. Since each dialogue has one annotated gold domain/tool, tool-set exact match measures routing, not genuinely compositional multi-tool ground truth; interpret multi-tool selection results accordingly.

## Recommended decision policy (conditional on observed results)
- Prefer `raw` if typo-case final EM/recall is statistically similar to corrected arms and time/cost is lower.
- Prefer `last` if it recovers typo-case routing/retrieval and downstream outcomes without substantial clean-case regressions.
- Prefer `full` only if multi-turn historical errors measurably improve task outcomes beyond `last`, exceeding additional latency and input-drift risk.
- Enable `react` selectively if 2nd/3rd search delivers meaningful final EM/grounded recall gains, and P95 remains acceptable.

License: refer to original MultiDoc2Dial dataset terms and model licenses.
