# Small paired pilot: clean versus synthetic typos

## At a glance

We completed **16 evaluations**: the same 8 dialogue cases were each run once with the original input (`clean`) and once with synthetic adjacent-letter transpositions (`typo`). There were 2 cases per domain (DMV, SSA, VA, StudentAid). All runs used `gpt-6-luna`, the same local retrieval models/index, `raw` correction (no separate spell-correction call), direct architecture, direct/no-ReAct agent, and a typo-aware router prompt. Thus this is a **paired pilot of typo input under one fixed system configuration**, not a comparison of correction strategies or agents. The table and interpretation below were calculated from the 8 matched clean/typo pairs.

| Metric | Clean mean (n=8) | Typo mean (n=8) | Mean paired change (typo − clean) |
|---|---:|---:|---:|
| Answer token F1 | 0.168 | 0.221 | +0.053 |
| Best-subspan EM (best support-span token F1) | 0.204 | 0.257 | +0.053 |
| Router exact match | 0.750 | 0.750 | 0.000 |
| Citation document F1 | 0.250 | 0.583 | +0.333 |
| Final document Recall@5 | 0.375 | 0.750 | +0.375 |
| Final span Recall@5 | 0.375 | 0.625 | +0.250 |
| Model latency per evaluation | 7.80 s | 7.53 s | −0.28 s |

Pairwise direction across the 8 cases: answer token F1 improved in 5 cases, declined in 2, and tied in 1; best-subspan score improved in 5 and declined in 3; router exact match was unchanged in all 8. Retrieval and citation means increased, but most paired cases tied on those metrics (the positive changes came from 2–3 cases). These are small-sample descriptive results, **not evidence that typos improve performance**.

## What was compared?

Each case appeared in both input conditions, keeping domain and underlying question fixed. The typo version applies deterministic adjacent-character swaps to eligible words in user turns; the agent's previous turns are unchanged. In this pilot, typo counts per case ranged from 1 to 7. The same `raw` + direct + typo-aware setup was kept for both versions. The model still generates answers independently, so generation variation and sensitivity to the exact question/context are part of the observed paired difference.

`raw` means no separate correction model call. The direct architecture makes one structured routing/query call, retrieves evidence, and generates the answer without ReAct follow-up searches. The typo-aware instruction asks the router to infer intended words; it does not separately correct the dialogue first.

## Metric meanings

- **Answer token F1:** word overlap between the generated reply and reference agent reply; partial overlap is credited, but paraphrases and wording differences affect it.
- **Best-subspan EM:** here this is a graded score from 0 to 1: the best token-F1 overlap between the generated reply and any annotated supporting document span. Higher means more shared words with at least one gold support passage; it is not a human judgement of correctness.
- **Router exact match:** whether the model selected exactly the annotated domain search tool. The dataset generally has one gold tool per case.
- **Citation document F1:** overlap between cited document IDs and gold document IDs.
- **Final document Recall@5:** share of gold documents present among the final five retrieved chunks.
- **Final span Recall@5:** share of annotated evidence spans covered in those final five chunks.
- **Latency:** model-call elapsed time recorded for each evaluation; retrieval runs locally. It is not API cost and can vary with service load/cache behavior.

## Fairness and limitations

- **The clean-versus-typo contrast is paired:** same 8 cases and fixed pipeline factors appear in both groups. This is more controlled than the prior mixed legacy snapshot.
- The pilot uses only **2 dialogues per domain / 8 total**. A single unusual or misrouted example changes the mean substantially; no significance test or broad generalization is warranted.
- The first 8 eligible IDs from the frozen sample were selected deterministically by the harness, not randomly sampled anew. This is reproducible but not a powered representative sample.
- The gold reference is free-form, and lexical-overlap scores are imperfect proxies for meaning. A high retrieval/citation score does not guarantee a correct response; a low overlap score does not prove the answer is wrong.
- The `clean` group is source dialogue without *new synthetic* transpositions; it can contain natural spelling/grammar errors already present in the dataset.
- Results only apply to `gpt-6-luna` under the direct architecture, typo-aware prompt, raw correction setting, and chosen local retrieval models. They say nothing fair about `last` vs. `full`, typo-awareness on vs. off, or ReAct vs. direct.
- The benchmark's `--fresh-run` summary step initially needed a small robustness fix to create its output directory. The 16 row-level results were saved before that fix; after the fix, summary files were regenerated from those same rows without repeating API calls.

## Run record

- Model: `gpt-6-luna`
- Evaluations: 16 (8 clean + 8 typo; 8 complete matched pairs)
- API calls recorded by the harness: 32 (2 per evaluation)
- Total reported model tokens: 36,348
- Sum of recorded API-call latency: 122.62 seconds (wall time for all 16 evaluations was longer due to local setup/retrieval)
- Output directory: local `results-small-paired/` (ignored by Git because it contains dialogue-level details)
- Seed: `20261020`; 2 cases per each of the 4 domains
- Rows were written in shuffled condition order, not always clean-first.

The token count is the best available usage/cost indicator in this run. Dollar cost is not reported because the exact billed rates/discounts for this model/account were not independently verified. Raw row/detail files are intentionally not committed because they include source dialogue content.
