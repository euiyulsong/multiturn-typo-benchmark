# Exploratory analysis of legacy results

**Status: exploratory only — not results for the current benchmark design.**

This report analyzes the local legacy `rows.jsonl` and `details.jsonl` snapshot: 973 paired rows across three correction strategies, two input conditions, and direct/ReAct agents. The rows combine an earlier legacy run and an incomplete/incorrect-factorial attempt; they do not correspond to the frozen 100-case factorial described in the repository README. The intended current main experiment remains **0/1,600 completed**, blocked by API quota. Do not use these figures as evidence for the current experiment or as a clean causal comparison.

## Answer scoring

Strict normalized response exact match is **0/973 (0.0%)**. This is a poor summary for free-form answers: a valid paraphrase can fail exact matching. We therefore report **best-subspan EM** alongside answer token F1.

For each prediction, best-subspan EM is the maximum token F1 between the answer and any annotated supporting source span for that case, using the benchmark's NFKC/lowercase/punctuation/article normalization. It is a graded overlap score in [0, 1], not a binary exact-match rate. The values below were recomputed from the JSONL prediction and gold span offsets against the downloaded MultiDoc2Dial source archive. They are not written back into the legacy raw files.

| Metric | Mean / count |
|---|---:|
| Strict response EM | 0.000 (0/973 exact matches) |
| Answer token F1 vs. gold agent reply | 0.246 |
| Best-subspan EM (best support-span token F1) | **0.263** |
| Best-subspan score ≥ 0.50 | 120/973 (12.3%) |
| Citation document F1 | 0.542 |
| Final document Recall@5 | 0.749 |
| Final span Recall@5 | 0.634 |
| Router exact match | 0.963 |
| Mean end-to-end latency | 12.9 s |

## By correction strategy

| Strategy | Rows | Answer token F1 | Best-subspan EM | Citation doc F1 | Final doc Recall@5 | Final span Recall@5 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 326 | 0.236 | 0.250 | 0.541 | 0.764 | 0.629 | 11.3 s |
| last | 330 | 0.253 | 0.272 | 0.507 | 0.718 | 0.613 | 13.8 s |
| full | 317 | 0.249 | 0.268 | 0.578 | 0.767 | 0.663 | 13.7 s |

`full` has the strongest average retrieval and citation scores in this legacy snapshot; `last` has the highest best-subspan score, with only a small difference from the other strategies. These are descriptive, unpaired aggregate comparisons across mixed legacy coverage, not reliable treatment effects.

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

The typo and clean groups have similar mean answer scores here; typo rows have lower citation F1 and slightly lower final document recall. SSA stands out for the weakest routing, citation, and best-subspan averages. Again, these are exploratory descriptions of the mixed snapshot.

## Interpretation

- The strict response EM of zero does **not** mean no prediction contains a correct answer: best-subspan EM averages 0.263, and 12.3% of predictions score at least 0.50 against an annotated support span.
- Retrieval finds a gold document in the final top five for about three quarters of rows, while answer overlap remains modest. This suggests answer grounding/generation deserves inspection in addition to retrieval, but does not alone establish a causal bottleneck.
- ReAct has about the same aggregate answer token F1 as direct (0.246 for each) and higher mean latency (15.8 s vs. 9.9 s) in the mixed data. This is not a controlled architecture comparison.
- The current experiment and its intended paired comparisons should be run before drawing conclusions about typo correction, prompt awareness, architecture, or ReAct.

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
