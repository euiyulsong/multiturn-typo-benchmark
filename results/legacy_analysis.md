# Exploratory analysis of legacy results

**Status: exploratory only — not results for the current benchmark design.**

This report analyzes the local legacy `rows.jsonl` and `details.jsonl` snapshot: 973 rows across three correction strategies, two input conditions, and direct/ReAct agents. The rows combine an earlier legacy run and an incomplete/incorrect-factorial attempt; they do not correspond to the frozen 100-case factorial described in the repository README. The intended current main experiment remains **0/1,600 completed**, blocked by API quota.

## Is this a fair comparison?

**No—not for deciding which strategy or agent is better.** These are useful descriptive summaries of the saved rows, but the legacy snapshot is not balanced or controlled well enough to support fair winner claims:

- Only **202/973** rows contain the `architecture` and `typo_aware` factors needed to separate those effects. The other 771 rows lack both; those missing values cannot safely be assumed to mean “staged” and “typo-aware off.”
- The 12 visible strategy × input × agent groups have different sizes (between 71 and 93 rows). The nominal design would have the same cases in each comparison group, but this snapshot is partial and incomplete.
- No case has all 12 visible combinations. For example, only **43–49 cases** have both clean and typo versions for a given strategy and agent. Treatment comparisons therefore do not consistently use the same examples.
- **75 rows repeat an already-present case/condition/strategy/agent/architecture/typo-awareness combination.** The repeated records may reflect the mixed run history and should not be counted as independent cases.
- The means below are unweighted averages over rows; they do not deduplicate repeats or pair outcomes case-by-case. Differences may reflect which cases or run variants are present, not the strategy itself. No uncertainty intervals or significance tests are reported here.

So the answer to “which one is better?” is **not established by these numbers**. A fair comparison requires the same preselected cases run in every compared condition, with architecture and prompt factors recorded, no duplicate treatment rows, and per-case paired deltas. That is the intent of the current experiment, which has not yet run because of API quota.

## What task is being measured?

Each benchmark item is a **multi-turn customer-support dialogue** from one of four domains: DMV, Social Security (SSA), Veterans Affairs (VA), or federal student aid. The model receives earlier dialogue turns plus a final user question, chooses a domain-specific search tool/query, retrieves passages from that domain's documents, and writes the next agent response. In the `typo` condition, synthetic adjacent-letter swaps are introduced in user turns; `clean` preserves the source dialogue. The saved legacy run contains `raw` (no separate correction), `last` (correct only the latest user message), and `full` (correct all user messages), as well as `direct` and `react` agent variants.

The target response is natural free-form text, while the annotations also identify relevant source documents and support spans. Thus, matching the entire reference sentence is stricter than checking whether the answer covers the supported information. The metrics below answer different questions and should not be treated as interchangeable.

## Metric guide and answer scoring

- **Rows / N:** number of saved model evaluations in that group. Repeated or unmatched rows mean this is not necessarily the number of unique, comparable examples.
- **Answer token F1:** word overlap between the model response and the dataset's reference agent reply. Precision penalizes extra words; recall penalizes missing reference words; F1 combines them. It gives partial credit to paraphrases but is still sensitive to wording.
- **Strict response EM:** 1 only if the normalized full model response matches the normalized full reference reply; otherwise 0. Because replies are free-form, a correct paraphrase can still score 0.
- **Best-subspan EM:** despite the “EM” name, the reported value here is a graded score, not a yes/no exact match. For each case, it is the **highest token F1** between the answer and any annotated supporting source span. A high score means strong lexical overlap with at least one gold span; it does not prove the full response is correct or fully grounded.
- **Citation document F1:** precision/recall F1 for the set of documents cited by the answer versus the annotated gold-document set. It measures source attribution, not factual correctness of every sentence.
- **Router EM:** whether the selected set of domain search tools exactly matches the gold tool set. Here, most examples have one gold domain tool, so this mainly tests routing to the right domain; it is not a broad test of complex multi-tool composition.
- **Document Recall@5:** whether the gold document(s) occur among the five final retrieved chunks, averaged across examples (with partial credit when there are multiple gold documents). It tests retrieval, not answer quality.
- **Span Recall@5:** whether annotated gold passage spans are covered by the final top-five retrieved evidence. A retrieved document may be correct while the specific passage is missing.
- **Latency:** mean elapsed wall-clock seconds per saved run. It can vary with caching, run conditions, and number of calls; it is not a clean efficiency comparison in this mixed snapshot.
- **`raw` / `last` / `full`:** no separate typo correction / correct only the latest user turn / correct all user turns. **`direct` / `react`:** answer after the initial search / may inspect results and search again (up to three searches in the intended design).

Strict normalized response exact match is **0/973 (0.0%)**. This is a poor summary for free-form answers: a valid paraphrase can fail exact matching. We therefore report **best-subspan EM** alongside answer token F1.

For each prediction, best-subspan EM is the maximum token F1 between the answer and any annotated supporting source span for that case, using the benchmark's NFKC/lowercase/punctuation/article normalization. It is a graded overlap score in [0, 1], not a binary exact-match rate. The values below were recomputed from the JSONL prediction and gold span offsets against the downloaded MultiDoc2Dial source archive. They are not written back into the legacy raw files.

| Metric | What it says | Mean / count |
|---|---|---:|
| Strict response EM | Full reply matches reference | 0.000 (0/973 exact matches) |
| Answer token F1 vs. gold agent reply | Word overlap with reference reply | 0.246 |
| Best-subspan EM (best support-span token F1) | Best word overlap with any gold evidence span | **0.263** |
| Best-subspan score ≥ 0.50 | At least 0.50 overlap with one gold span | 120/973 (12.3%) |
| Citation document F1 | Quality of cited document set | 0.542 |
| Final document Recall@5 | Gold document retrieved in final top five | 0.749 |
| Final span Recall@5 | Gold evidence span covered in final top five | 0.634 |
| Router exact match | Correct gold tool/domain set selected | 0.963 |
| Mean end-to-end latency | Average elapsed seconds per run | 12.9 s |

## By correction strategy

| Strategy | Rows | Answer token F1 | Best-subspan EM | Citation doc F1 | Final doc Recall@5 | Final span Recall@5 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 326 | 0.236 | 0.250 | 0.541 | 0.764 | 0.629 | 11.3 s |
| last | 330 | 0.253 | 0.272 | 0.507 | 0.718 | 0.613 | 13.8 s |
| full | 317 | 0.249 | 0.268 | 0.578 | 0.767 | 0.663 | 13.7 s |

In the raw averages, `full` has the strongest retrieval and citation scores, while `last` has the highest best-subspan score. The score differences are small, and because the groups are unbalanced, partly duplicated, and not reliably paired, **these rankings are not fair estimates of which correction strategy is better**. Also, in the intended design, direct architecture bypasses separate correction calls, so `raw`/`last`/`full` are not meaningful distinct treatments for direct rows.

## By input condition and domain

| Group | Rows | Answer token F1 | Best-subspan EM | Citation doc F1 | Final doc Recall@5 |
|---|---:|---:|---:|---:|---:|
| clean | 479 | 0.242 | 0.263 | 0.561 | 0.756 |
| typo | 494 | 0.250 | 0.264 | 0.523 | 0.743 |

| Domain | Rows | Best-subspan EM | Citation doc F1 | Final doc Recall@5 | Router EM |
|---|---:|---:|---:|---:|---:|
| StudentAid | 262 | 0.271 | 0.641 | 0.798 | 1.000 |
| SSA | 240 | 0.232 | 0.334 | 0.713 | 0.921 |
| DMV | 236 | 0.272 | 0.693 | 0.788 | 0.953 |
| VA | 235 | 0.277 | 0.491 | 0.694 | 0.974 |

The typo and clean groups have similar raw mean answer scores here; typo rows have lower citation F1 and slightly lower final document recall. That does **not** show that typos have no effect: the conditions are not fully paired and the row coverage differs. SSA has the weakest observed routing, citation, and best-subspan averages. Domain differences may also reflect case composition and missing run factors, so they are diagnostic clues rather than fair domain-vs-domain rankings.

## Interpretation

- Strict response EM of zero does **not** mean every answer is empty or entirely wrong: best-subspan EM averages 0.263, and 12.3% of rows score at least 0.50 against one annotated support span. These overlap values are not a human correctness judgment.
- Retrieval returns a gold document in the final top five on average in about three quarters of rows, while response overlap is modest. This flags answer grounding/generation and retrieval evidence for further case review; it does not by itself prove which component causes the errors.
- ReAct and direct have the same aggregate answer token F1 (0.246); mean latency is 15.8 s for ReAct and 9.9 s for direct. Since architecture factors/case coverage are mixed, this is **not** a controlled estimate of the cost or benefit of ReAct.
- A meaningful “problem” to investigate is the gap between finding the right document and producing a well-supported next response, especially for SSA cases with lower observed router/citation scores. Verify this by reading paired examples and using the complete experiment, not by treating the aggregate as a diagnosis.
- The current experiment and its complete, paired comparisons should be run before drawing conclusions about typo correction, prompt awareness, architecture, or ReAct.

## Reproducibility

The source rows/details and dataset archive used for this analysis are local and intentionally excluded from Git. The script below verifies the 973-row aggregates without publishing conversation text or benchmark row data:

```bash
python3 - <<'PY'
import json, pathlib, re, statistics, unicodedata, zipfile
from collections import Counter

out = pathlib.Path("results")
docs = json.loads(zipfile.ZipFile(out / "data/multidoc2dial.zip").read(
    "multidoc2dial/multidoc2dial_doc.json"))["doc_data"]

def norm(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def token_f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    return 2 * sum((Counter(p) & Counter(g)).values()) / (len(p) + len(g))

rows = [json.loads(line) for line in (out / "rows.jsonl").open()]
details = [json.loads(line) for line in (out / "details.jsonl").open()]
for row, detail in zip(rows, details):
    answer = detail["pred_answer"].get("answer", "")
    spans = [docs[s["domain"]][s["doc_id"]]["doc_text"][s["start"]:s["end"]]
             for s in detail["gold_spans"]]
    row["best_subspan_em"] = max(token_f1(answer, span) for span in spans)

for metric in ("answer_em", "answer_token_f1", "best_subspan_em",
               "citation_doc_f1", "final_doc_recall@5", "final_span_recall@5",
               "router_em", "latency_s"):
    print(metric, round(statistics.mean(r[metric] for r in rows), 4))
print("best_subspan_em >= 0.50", sum(r["best_subspan_em"] >= .5 for r in rows), "/", len(rows))
PY
```
