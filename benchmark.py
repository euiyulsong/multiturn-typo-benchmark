#!/usr/bin/env python3
"""MultiDoc2Dial paired typo × correction × direct/ReAct (<=3 search steps) benchmark.

This is an experiment HARNESS, not a precomputed benchmark. No numbers are fabricated.
Install requirements and provide OPENAI_API_KEY + BENCH_MODEL to run online.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import time
import unicodedata
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"
DOMAINS = ("dmv", "ssa", "va", "studentaid")
DOMAIN_EXPLAIN = {
    "dmv": "driver's license, registration, state motor vehicles",
    "ssa": "Social Security, retirement, benefits, disability",
    "va": "US Veterans Affairs and veterans benefits",
    "studentaid": "federal student aid, FAFSA, grants and loans",
}
STOPWORDS = set("about above after again all also any are because before being between can could does each from have into more most other over some such than that them then there these they this those through under until very were what when where which while will with would your please thanks thank hello hi need want tell help just like know".split())


def stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:15], 16)


def norm(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str, gold: str) -> float:
    return float(norm(pred) == norm(gold))


def token_f1(pred: str, gold: str) -> float:
    a, b = norm(pred).split(), norm(gold).split()
    if not a or not b:
        return float(a == b)
    shared = sum((Counter(a) & Counter(b)).values())
    return (2 * shared / (len(a) + len(b))) if shared else 0.0


def subspan_em(pred: str, spans: list[str], gold_answer: str = "") -> float:
    """Fraction of annotated support spans found, with a short-answer fallback."""
    p = norm(pred)
    normalized = [norm(s) for s in spans if norm(s)]
    candidates = [s for s in normalized if s in p] if normalized else []
    if normalized:
        # Dataset reference spans may be long. Use token coverage against each
        # annotated support span, then exact-match the factoid answer as fallback.
        return max((token_f1(pred, span) for span in spans if norm(span)), default=0.0)
    gold = norm(gold_answer)
    return float(bool(gold) and gold in p)


def set_f1(pred: list[str], gold: list[str]) -> float:
    p, g = set(pred), set(gold)
    if not p or not g:
        return float(p == g)
    return 2 * len(p & g) / (len(p) + len(g))


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_data(data_dir: Path, zip_path: str | None) -> tuple[dict, dict]:
    data_dir.mkdir(parents=True, exist_ok=True)
    p = Path(zip_path) if zip_path else data_dir / "multidoc2dial.zip"
    if not p.exists():
        print("Downloading official MultiDoc2Dial dataset:", DATASET_URL, flush=True)
        urllib.request.urlretrieve(DATASET_URL, p)
    with zipfile.ZipFile(p) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX/")]
        def load(suffix: str) -> dict:
            matches = [n for n in names if n.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {suffix}, found {matches}. ZIP files: {names[:30]}")
            return json.loads(z.read(matches[0]).decode("utf-8-sig"))
        docs = load("multidoc2dial_doc.json")["doc_data"]
        dial = load("multidoc2dial_dial_validation.json")["dial_data"]
    return docs, dial


def build_cases(doc_data: dict, dial_data: dict, seed: int, n_per_domain: int = 25) -> list[dict]:
    cases = []
    for domain in DOMAINS:
        dialogs = dial_data.get(domain, [])
        eligible = []
        for dialog in dialogs:
            turns = dialog.get("turns", [])
            choices = []
            for i, turn in enumerate(turns[:-1]):
                if turn.get("role") != "user" or sum(t.get("role") == "user" for t in turns[:i]) < 1:
                    continue
                nxt = turns[i+1]
                if nxt.get("role") != "agent":
                    continue
                references = [r for r in nxt.get("references", []) if r.get("label") in ("solution", "precondition")]
                if not references:
                    continue
                gold_docs = sorted({str(r.get("doc_id", "")) for r in references if r.get("doc_id")})
                if not gold_docs:
                    continue
                # Require all referenced documents and spans to exist in the raw corpus.
                if any(docid not in doc_data.get(domain, {}) for docid in gold_docs):
                    continue
                valid_refs = []
                for r in references:
                    doc = doc_data[domain][r["doc_id"]]
                    sp = doc.get("spans", {}).get(str(r["id_sp"]))
                    if sp:
                        doc_text = doc.get("doc_text", "")
                        valid_refs.append({"doc_id": r["doc_id"], "id_sp": str(r["id_sp"]),
                                            "start": int(sp["start_sp"]), "end": int(sp["end_sp"]),
                                            "domain": domain,
                                            "text": doc_text[int(sp["start_sp"]):int(sp["end_sp"])]})
                if len(valid_refs) != len(references):
                    continue
                history = [{"role": t["role"], "text": t["utterance"]} for t in turns[:i]]
                choices.append({"id": f'{dialog["dial_id"]}:{turn["turn_id"]}',
                                "dialogue_id": dialog["dial_id"], "domain": domain,
                                "history": history, "question": turn["utterance"],
                                "gold_answer": nxt["utterance"], "gold_docs": gold_docs,
                                "gold_spans": valid_refs,
                                "gold_tools": [f"search_{domain}"]})
            if choices:
                # One target turn per dialogue; fixed deterministic selection.
                idx = stable_seed(f'{seed}|{dialog["dial_id"]}') % len(choices)
                eligible.append(choices[idx])
        eligible.sort(key=lambda x: stable_seed(f'{seed}|{x["id"]}'))
        if len(eligible) < n_per_domain:
            raise RuntimeError(f"Not enough qualified {domain} validation dialogues: {len(eligible)} < {n_per_domain}")
        cases.extend(eligible[:n_per_domain])
    cases.sort(key=lambda x: x["id"])
    return cases


def corrupt_one(text: str, rng: random.Random, typo_types: tuple[str, ...] = ("transpose",),
                edits_per_word: int = 1) -> tuple[str, dict | None]:
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z]{5,}", text)
             if m.group().lower() not in STOPWORDS]
    if not words:
        words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z]{4,}", text)]
    if not words:
        return text, None
    start, end, word = rng.choice(words)
    # One deterministic edit in a sampled user word; labels and references stay untouched.
    available = [kind for kind in typo_types if kind in {"transpose", "delete", "insert", "replace"}]
    if not available:
        raise ValueError(f"No supported typo types in {typo_types}")
    typo = word
    applied=[]
    for _ in range(max(1,edits_per_word)):
        kind = rng.choice(available)
        if kind == "transpose":
            positions = [i for i in range(len(typo)-1) if typo[i].lower() != typo[i+1].lower()]
            if not positions:
                continue
            j = rng.choice(positions)
            typo = typo[:j] + typo[j+1] + typo[j] + typo[j+2:]
        elif kind == "delete":
            if len(typo) <= 2:
                continue
            j = rng.randrange(len(typo))
            typo = typo[:j] + typo[j+1:]
        elif kind == "insert":
            j = rng.randrange(len(typo)+1)
            letter = rng.choice("abcdefghijklmnopqrstuvwxyz")
            typo = typo[:j] + letter + typo[j:]
        else:  # replace
            j = rng.randrange(len(typo))
            choices = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != typo[j].lower()]
            letter = rng.choice(choices)
            if typo[j].isupper():
                letter = letter.upper()
            typo = typo[:j] + letter + typo[j+1:]
        applied.append(kind)
    if typo == word:
        return text, None
    return text[:start] + typo + text[end:], {"before": word, "after": typo, "position": start,
                                               "type": applied[0] if len(applied)==1 else "multi",
                                               "operations": applied}


def input_variant(case: dict, typo: bool, seed: int,
                  typo_types: tuple[str, ...] = ("transpose",),
                  edits_per_word: int = 1) -> tuple[list[dict], str, list[dict]]:
    history = [dict(m) for m in case["history"]]
    question = case["question"]
    edits = []
    if typo:
        for i, message in enumerate(history):
            if message["role"] == "user":
                rng = random.Random(stable_seed(f'{seed}|{case["id"]}|{i}'))
                message["text"], edit = corrupt_one(message["text"], rng, typo_types, edits_per_word)
                if edit:
                    edits.append({"turn": i, **edit})
        question, edit = corrupt_one(question, random.Random(stable_seed(f'{seed}|{case["id"]}|target')),
                                     typo_types, edits_per_word)
        if edit:
            edits.append({"turn": "target", **edit})
    return history, question, edits


def schema_object(name: str, fields: dict) -> dict:
    return {"type": "json_schema", "name": name, "strict": True, "schema": {
        "type": "object", "properties": fields, "required": list(fields),
        "additionalProperties": False}}


def string_schema() -> dict:
    return {"type": "string"}


class JsonModel:
    def __init__(self, model: str, cache_file: Path):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(max_retries=2, timeout=120)
        self.cache_file = cache_file
        self.cache: dict[str, dict] = {}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["cache_key"]] = row

    def ask(self, system: str, prompt: str, schema: dict) -> tuple[dict, float, int]:
        sig = json.dumps([self.model, system, prompt, schema], ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(sig.encode()).hexdigest()
        if key in self.cache:
            row = self.cache[key]
            return row["result"], row["latency_s"], row["tokens"]
        st = time.perf_counter()
        # Responses API with strict JSON schema; no unsupported temperature/seed parameter.
        try:
            result = self.client.responses.create(
                model=self.model,
                instructions=system,
                input=prompt,
                text={"format": schema},
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429 or "insufficient_quota" in str(exc):
                raise RuntimeError(
                    "Model API quota/credits are exhausted. Add API credits, then rerun the same command; "
                    "completed row-level results are cached and skipped.") from exc
            raise
        latency = time.perf_counter() - st
        if not result.output_text:
            raise RuntimeError(f"No text from {self.model}: {result.status}")
        obj = json.loads(result.output_text)
        tokens = int(getattr(getattr(result, "usage", None), "total_tokens", 0) or 0)
        row = {"cache_key": key, "result": obj, "latency_s": latency, "tokens": tokens}
        self.cache[key] = row
        with self.cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return obj, latency, tokens


def fix_input(llm: JsonModel, mode: str, history: list[dict], question: str) -> tuple[list[dict], str, dict]:
    if mode == "raw":
        return history, question, {"seconds": 0.0, "calls": 0, "tokens": 0}
    system = ("You perform only minimal spelling/typo correction in English user utterances. "
              "Preserve intent, entities, numbers, coreference, dialogue structure and wording. "
              "Never answer the question, expand references, summarize or paraphrase. "
              "Return a JSON object matching the required schema. Agent utterances are read-only.")
    if mode == "last":
        prompt = json.dumps({"history_context": history, "last_user_utterance_to_fix": question}, ensure_ascii=False)
        result, sec, tokens = llm.ask(system + " Edit ONLY the last user utterance. Output one line.", prompt,
            schema_object("fix_last", {"text": string_schema()}))
        cleaned = " ".join(result["text"].splitlines())
        return history, cleaned, {"seconds": sec, "calls": 1, "tokens": tokens}
    if mode == "full":
        user_indices = [i for i, msg in enumerate(history) if msg["role"] == "user"]
        utterances = [history[i]["text"] for i in user_indices] + [question]
        prompt = json.dumps({"dialogue_context": history,
                             "user_utterances_to_fix_in_order": utterances}, ensure_ascii=False)
        result, sec, tokens = llm.ask(system + " Output the same number of utterances in the same order.", prompt,
            schema_object("fix_full", {"utterances": {"type": "array", "items": string_schema()}}))
        out = result["utterances"]
        if len(out) != len(utterances):
            raise RuntimeError(f"Full correction returned {len(out)} utterances, expected {len(utterances)}")
        corrected = [dict(msg) for msg in history]
        for i, value in zip(user_indices, out[:-1]):
            corrected[i]["text"] = " ".join(value.splitlines())
        return corrected, " ".join(out[-1].splitlines()), {"seconds": sec, "calls": 1, "tokens": tokens}
    raise ValueError(mode)


def rewrite_query(llm: JsonModel, history: list[dict], question: str,
                  typo_aware: bool, one_sentence: bool = False) -> tuple[str, dict]:
    """Rewrite the latest question before the separate tool-selection call."""
    system = ("Rewrite the latest user question into a concise standalone search query. "
              "Use dialogue context only to resolve references and ellipsis. Preserve intent and details. "
              "Do not select tools, answer the question, or introduce facts. Return JSON with one string field named query.")
    if typo_aware:
        system += (" The dialogue may contain spelling mistakes or typos. Infer intended words from context "
                   "and correct them in the standalone query without changing meaning.")
    if one_sentence:
        system += " Return exactly one concise sentence. Do not include a list, preamble, or multiple alternative queries."
    result, sec, tokens = llm.ask(system, format_dialogue(history, question),
                                  schema_object("rewrite_query", {"query": string_schema()}))
    return result["query"].strip() or question, {"seconds": sec, "calls": 1, "tokens": tokens}


@dataclass
class Chunk:
    id: str
    doc_id: str
    domain: str
    text: str
    start: int
    end: int


def build_chunks(doc_data: dict, size: int = 1050, overlap: int = 180) -> list[Chunk]:
    chunks = []
    for domain in DOMAINS:
        for docid in sorted(doc_data[domain]):
            doc = doc_data[domain][docid]
            content = doc.get("doc_text", "")
            title = doc.get("title", "")
            begin = 0
            i = 0
            while begin < len(content):
                end = min(begin + size, len(content))
                if end < len(content):
                    pos = content.rfind(" ", max(begin+size//2, end-120), end)
                    if pos > begin:
                        end = pos
                passage = f"{title}\n{content[begin:end]}"
                chunks.append(Chunk(f"{domain}:{hashlib.sha1(docid.encode()).hexdigest()[:10]}:{i}",
                                    docid, domain, passage, begin, end))
                if end >= len(content):
                    break
                begin = max(begin+1, end-overlap)
                i += 1
    return chunks


class Retrieval:
    def __init__(self, chunks: list[Chunk], dense_model: str, cross_model: str, device: str,
                 source_docs: dict | None = None, batch_size: int = 64):
        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.np = np
        self.chunks = {c.id: c for c in chunks}
        self.source_docs = source_docs or {}
        self.domains = {}
        self.bm25 = {}
        self.embs = {}
        self.encoder = SentenceTransformer(dense_model, device=device)
        self.cross = CrossEncoder(cross_model, device=device)
        for domain in DOMAINS:
            section = [c for c in chunks if c.domain == domain]
            self.domains[domain] = section
            self.bm25[domain] = BM25Okapi([self.tokenize(c.text) for c in section])
            arr = self.encoder.encode([c.text for c in section], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
            self.embs[domain] = arr

    @staticmethod
    def tokenize(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    def search(self, domain: str, query: str, per_branch: int = 30, rerank_pool: int = 20,
               output_k: int = 5) -> dict:
        import numpy as np
        sec = time.perf_counter()
        docs = self.domains[domain]
        toks = self.tokenize(query)
        bm = np.asarray(self.bm25[domain].get_scores(toks))
        bm_idx = np.argsort(-bm, kind="stable")[:per_branch].tolist()
        # Do not cache query encodings: each query pays the same online embedding cost.
        qvec = self.encoder.encode([query], normalize_embeddings=True,
                                   convert_to_numpy=True)[0]
        sim = self.embs[domain] @ qvec
        dense_idx = np.argsort(-sim, kind="stable")[:per_branch].tolist()
        fusion = defaultdict(float)
        for ids in (bm_idx, dense_idx):
            for rank, idx in enumerate(ids, 1):
                fusion[idx] += 1.0/(60 + rank)
        rrf_idx = sorted(fusion, key=lambda k: (-fusion[k], k))[:rerank_pool]
        candidates = [(query, docs[i].text) for i in rrf_idx]
        scores = self.cross.predict(candidates, show_progress_bar=False) if candidates else []
        ce_idx = [idx for _,idx in sorted(zip(scores,rrf_idx),key=lambda z:-float(z[0]))]
        ce_scores = {docs[i].id: float(score) for score,i in zip(scores,rrf_idx)}
        duration = time.perf_counter()-sec
        return {"domain": domain, "query": query,
                "bm25_ids": [docs[i].id for i in bm_idx],
                "dense_ids": [docs[i].id for i in dense_idx],
                "rrf_ids": [docs[i].id for i in rrf_idx],
                "ce_ids": [docs[i].id for i in ce_idx],
                "top_ids": [docs[i].id for i in ce_idx[:output_k]],
                "ce_scores": ce_scores, "seconds": duration}

    def run_tools(self, tools: list[str], query: str) -> dict:
        steps = []
        for tool in tools:
            domain = tool.removeprefix("search_")
            if domain not in DOMAINS:
                continue
            steps.append(self.search(domain, query))
        def aggregate(key: str) -> list[str]:
            # Fuse ranked lists from distinct tools with RRF to avoid one tool dominating.
            scores = defaultdict(float)
            for s in steps:
                for rank, chunk_id in enumerate(s[key], 1):
                    scores[chunk_id] += 1/(60 + rank)
            return sorted(scores, key=lambda x:(-scores[x], x))
        return {"searches": steps,
                "bm25_ids": aggregate("bm25_ids"),
                "dense_ids": aggregate("dense_ids"),
                "rrf_ids": aggregate("rrf_ids"),
                "ce_ids": aggregate("ce_ids"),
                "top_ids": aggregate("top_ids"),
                "seconds": sum(s["seconds"] for s in steps)}

    def search_gold_domain(self, domain: str, query: str) -> dict:
        """Run retrieval against the annotated domain, without an LLM tool router."""
        result=self.search(domain,query)
        result["searches"]=[{k:v for k,v in result.items() if k!="searches"}]
        for key in ("bm25_ids","dense_ids","rrf_ids","ce_ids","top_ids"):
            result[key]=result[key]
        return result


def format_dialogue(history: list[dict], question: str) -> str:
    messages = history + [{"role": "user", "text": question}]
    return "\n".join(f'{m["role"].upper()}: {m["text"]}' for m in messages)


def route(llm: JsonModel, history: list[dict], question: str,
          typo_aware: bool = False) -> tuple[list[str], str, dict]:
    tool_names = [f"search_{d}" for d in DOMAINS]
    schema = schema_object("multi_tool_route", {
        "tools": {"type": "array", "items": {"type": "string", "enum": tool_names}},
        "query": string_schema()})
    typo_instruction = (" The input may contain spelling errors or typos: infer the intended words from dialogue context, "
                        "and produce a corrected, focused standalone retrieval query without changing intent."
                        if typo_aware else "")
    sys = ("You are a multi-tool search router. Pick one or more useful tools to answer the final user question. "
           "Use full dialogue for context. Select no more than two, deduplicate, and order by importance. "
           "Search tools: " + json.dumps(DOMAIN_EXPLAIN) + ". "
           "Write a focused standalone retrieval query resolving references only as needed. "
           "Return JSON with tools and query. Never use dataset labels." + typo_instruction)
    obj, latency, tokens = llm.ask(sys, format_dialogue(history, question), schema)
    tools = list(dict.fromkeys(t for t in obj["tools"] if t in tool_names))[:2]
    if not tools:
        tools = ["search_dmv"]  # invalid/empty selection recorded separately as router failure
    query = obj["query"].strip() or question
    return tools, query, {"seconds": latency, "calls": 1, "tokens": tokens,
                         "router_raw": obj, "selection_query": query}


def evidence_block(ret: Retrieval, ids: list[str], n: int = 5) -> str:
    parts = []
    for i, cid in enumerate(ids[:n], 1):
        c = ret.chunks[cid]
        parts.append(f"[{i}] chunk_id={cid} | doc_id={c.doc_id} | tool=search_{c.domain}\n{c.text[:1200]}")
    return "\n\n".join(parts)


def decide_react(llm: JsonModel, history: list[dict], question: str, ret: Retrieval,
                 all_ids: list[str], tool_history: list[dict]) -> tuple[dict, dict]:
    schema = schema_object("react_decision", {
        "action": {"type": "string", "enum": ["finish", "search"]},
        "tool": {"type": "string", "enum": [f"search_{d}" for d in DOMAINS]},
        "query": string_schema()})
    system = ("You are a search agent. Decide if currently retrieved evidence is sufficient to answer "
              "the final user question accurately. If not, choose ONE useful next search tool and a focused "
              "query (a new facet or clearer query). Do not access gold labels. "
              "Return JSON. Use action=finish if evidence is sufficient or nothing useful remains to search.")
    obj, sec, tokens = llm.ask(system, json.dumps({"dialogue": format_dialogue(history, question),
          "search_history": tool_history, "evidence": evidence_block(ret, all_ids, 6)},ensure_ascii=False),schema)
    return obj, {"seconds": sec, "calls": 1, "tokens": tokens}


def answer(llm: JsonModel, history: list[dict], question: str,
           ret: Retrieval, ids: list[str]) -> tuple[dict, dict]:
    schema = schema_object("answer_with_citations", {
        "answer": string_schema(),
        "evidence_doc_ids": {"type": "array", "items": string_schema()}})
    system = ("Answer the final user query as the next support agent response grounded ONLY in retrieved snippets. "
              "Be concise and faithful. If evidence is insufficient, state that. "
              "Cite only exact doc_id values from snippets. Return JSON only. "
              "Never use hidden gold/reference responses.")
    obj, sec, tokens = llm.ask(system, json.dumps({"dialogue": format_dialogue(history, question),
                        "retrieved_evidence": evidence_block(ret,ids,5)},ensure_ascii=False),schema)
    citations = list(dict.fromkeys(obj["evidence_doc_ids"]))
    allowed = {ret.chunks[x].doc_id for x in ids[:5]}
    obj["valid_evidence_doc_ids"] = [x for x in citations if x in allowed]
    obj["hallucinated_citations"] = [x for x in citations if x not in allowed]
    return obj, {"seconds": sec, "calls": 1, "tokens": tokens}


def relevance(chunks: dict[str, Chunk], ids: list[str], case: dict, k: int) -> dict:
    subset = [chunks[x] for x in ids[:k]]
    got_docs = {x.doc_id for x in subset}
    gold_docs = set(case["gold_docs"])
    hits = []
    for span in case["gold_spans"]:
        length = max(1, span["end"] - span["start"])
        hits.append(any(c.doc_id == span["doc_id"] and
                        max(0, min(c.end, span["end"]) - max(c.start, span["start"])) >= min(length, max(20, .5*length))
                        for c in subset))
    return {"doc_recall": len(got_docs & gold_docs)/len(gold_docs),
            "all_docs": float(gold_docs <= got_docs),
            "span_recall": sum(hits)/len(hits) if hits else 0.,
            "all_spans": float(all(hits)),
            "subspan_em": float(sum(hits)/len(hits)) if hits else 0.0,
            "all_subspan_em": float(all(hits))}


def mean_reciprocal_rank(chunks: dict[str, Chunk], ids: list[str], gold_docs: list[str], k: int) -> float:
    gold = set(gold_docs)
    return next((1.0 / rank for rank, cid in enumerate(ids[:k], 1)
                 if chunks[cid].doc_id in gold), 0.0)


def rrf_across_steps(step_results: list[dict], key: str) -> list[str]:
    # Each call's final rankings are fused. This is more comparable across different queries
    # than taking the maximum uncalibrated cross-encoder logit.
    agg = defaultdict(float)
    for r in step_results:
        for i, cid in enumerate(r[key], 1):
            agg[cid] += 1/(60+i)
    return sorted(agg, key=lambda c:(-agg[c],c))


def evaluate_one(case: dict, input_condition: str, strategy: str, agent: str,
                 llm: JsonModel, ret: Retrieval, seed: int,
                 architecture: str = "staged", typo_aware: bool = False,
                 retrieval_only: bool = False,
                 typo_types: tuple[str, ...] = ("transpose",),
                 skip_router: bool = False,
                 edits_per_word: int = 1,
                 rewrite_only_search: bool = False) -> tuple[dict, dict]:
    history, question, edits = input_variant(case, input_condition == "typo", seed,
                                             typo_types,edits_per_word)
    rewrite_history = [m for m in history if m["role"] == "user"]
    corrected_history, corrected_question, prep = fix_input(llm, strategy,history,question)
    if skip_router:
        corrected_history, corrected_question = history, question
        prep = {"seconds":0.0,"calls":0,"tokens":0}
        tools = [f"search_{case['domain']}"]
        query = format_dialogue(rewrite_history, corrected_question)
        router = {"seconds":0.0,"calls":0,"tokens":0,"router_raw":{"tools":tools},
                  "selection_query":query}
    elif architecture == "direct":
        # Joint tool-selection/query-writing call; no separate correction stage.
        corrected_history, corrected_question = history, question
        prep = {"seconds": 0.0, "calls": 0, "tokens": 0}
        tools, query, router = route(llm, rewrite_history, corrected_question,
                                     typo_aware=typo_aware)
    else:
        query, rewrite_cost = rewrite_query(llm, rewrite_history, corrected_question,
                                            typo_aware, one_sentence=rewrite_only_search)
        if rewrite_only_search:
            tools, _, router = route(llm, rewrite_history, query, typo_aware=typo_aware)
            router["selection_query"] = query
            router["router_raw"]["query"] = query
        else:
            tools, _, router = route(llm, rewrite_history, query, typo_aware=typo_aware)
        prep = {"seconds": prep["seconds"] + rewrite_cost["seconds"],
                "calls": prep["calls"] + rewrite_cost["calls"],
                "tokens": prep["tokens"] + rewrite_cost["tokens"]}
    # Fairness: every direct/ReAct experiment uses identical first search call.
    search_query = router["selection_query"] if architecture == "direct" else query
    first = (ret.search_gold_domain(case["domain"],search_query)
             if skip_router else ret.run_tools(tools,search_query))
    if retrieval_only:
        initial_predicted = router["router_raw"]["tools"]
        valid_predicted = list(dict.fromkeys(
            x for x in initial_predicted
            if x.startswith("search_") and x.removeprefix("search_") in DOMAINS))
        gold_tools = case["gold_tools"]
        if skip_router:
            valid_predicted=[]
            initial_predicted=[]
        metrics = {"case_id":case["id"], "domain":case["domain"],
                   "condition":input_condition, "strategy":strategy, "agent":agent,
                   "architecture":architecture, "typo_aware":typo_aware,
                   "typos_inserted":len(edits),
                   "router_skipped":bool(skip_router),
                   "rewrite_only_search":bool(rewrite_only_search),
                   "router_em":None if skip_router else float(set(valid_predicted)==set(gold_tools) and len(valid_predicted)==len(set(valid_predicted))),
                   "router_tool_f1":None if skip_router else set_f1(valid_predicted,gold_tools),
                   "router_selected_n":None if skip_router else len(initial_predicted),
                   "search_steps":1, "llm_calls":prep["calls"]+router["calls"],
                   "llm_tokens":prep["tokens"]+router["tokens"],
                   "latency_s":round(prep["seconds"]+router["seconds"],4),
                   "preprocess_latency_s":round(prep["seconds"],4),
                   "route_latency_s":round(router["seconds"],4),
                   "retrieval_latency_s":round(first["seconds"],4),
                   "reasoning_latency_s":0.0, "generation_latency_s":0.0}
        for stage in ("bm25", "dense", "rrf", "ce"):
            ids=first[stage+"_ids"]
            metrics[f"{stage}_mrr@10"]=mean_reciprocal_rank(ret.chunks,ids,case["gold_docs"],10)
            for k in (1,3,5,10):
                for name,value in relevance(ret.chunks,ids,case,k).items():
                    metrics[f"{stage}_{name}@{k}"]=value
        detail={"case_id":case["id"],"condition":input_condition,"strategy":strategy,
                "agent":agent,"architecture":architecture,"typo_aware":typo_aware,
                "edits":edits,"original_history":case["history"],"original_question":case["question"],
                "variant_history":history,"variant_question":question,
                "corrected_history":corrected_history,"corrected_question":corrected_question,
                "gold_tools":gold_tools,"gold_docs":case["gold_docs"],"gold_spans":case["gold_spans"],
                "predicted_tools":None if skip_router else initial_predicted,"router_query":query,
                "selection_and_search_query":search_query,
                "search_history":[{"tools":tools,"query":search_query}],
                "search_results":[{k:v for k,v in first.items() if k!="searches"}],
                "router_skipped":bool(skip_router),"rewrite_only_search":bool(rewrite_only_search),
                "metrics":metrics}
        return metrics,detail
    searched = [{"tools": tools,"query": search_query}]
    step_results = [first]
    extra_llm_sec = 0.0
    extra_llm_calls = 0
    extra_tokens = 0
    intermediate = []
    for step in range(2,4):
        if agent != "react":
            break
        available_ids = rrf_across_steps(step_results,"ce_ids")
        decision, cost = decide_react(llm,corrected_history,corrected_question,ret,available_ids,searched)
        extra_llm_sec += cost["seconds"]
        extra_llm_calls += cost["calls"]
        extra_tokens += cost["tokens"]
        intermediate.append({"decision_after_step":step-1,"decision":decision})
        if decision["action"] != "search":
            break
        next_tool = decision["tool"]
        next_query = decision["query"].strip() or corrected_question
        step_results.append(ret.run_tools([next_tool],next_query))
        searched.append({"tools":[next_tool],"query":next_query})
    final_ids = rrf_across_steps(step_results,"ce_ids")
    generated, gen_cost = answer(llm,corrected_history,corrected_question,ret,final_ids[:5])
    gold_subspans = []
    for span in case["gold_spans"]:
        document_text = ret.source_docs[span["domain"]][span["doc_id"]]["doc_text"]
        gold_subspans.append(document_text[span["start"]:span["end"]])
    stages = {}
    for stage in ("bm25", "dense", "rrf", "ce"):
        # First-step ranking, to measure impacts before ReAct independently.
        ids = first[stage+"_ids"]
        stages[f"{stage}_mrr@10"] = mean_reciprocal_rank(ret.chunks, ids, case["gold_docs"], 10)
        for k in (1,3,5,10):
            m = relevance(ret.chunks,ids,case,k)
            stages.update({f"{stage}_{name}@{k}":value for name,value in m.items()})
    for k in (1,3,5):
        m = relevance(ret.chunks,final_ids,case,k)
        stages.update({f"final_{name}@{k}":v for name,v in m.items()})
    retrieved_subspan_hits = []
    for span, span_text in zip(case["gold_spans"], gold_subspans):
        exact = bool(span_text) and any(
            ret.chunks[cid].doc_id == span["doc_id"] and
            norm(span_text) in norm(ret.chunks[cid].text)
            for cid in final_ids[:5])
        retrieved_subspan_hits.append(float(exact))
    stages["final_subspan_em@5"] = statistics.fmean(retrieved_subspan_hits) if retrieved_subspan_hits else 0.0
    stages["final_all_subspan_em@5"] = float(bool(retrieved_subspan_hits) and all(retrieved_subspan_hits))
    # Offline checkpoint probes: answer quality after ReAct search step 1/2/3.
    # NOT visible to agent and NOT counted in production API latency/calls.
    # Only the last probe is typically cached from the actual final answer generation.
    checkpoints = []
    offline_probe_calls = 0
    for step_count in range(1, len(step_results)+1):
        step_ids = rrf_across_steps(step_results[:step_count], "ce_ids")
        r = relevance(ret.chunks, step_ids, case, 5)
        checkpoint = {f"checkpoint_{k}@5_step{step_count}":v for k,v in r.items()}
        checkpoint[f"checkpoint_mrr@10_step{step_count}"] = mean_reciprocal_rank(ret.chunks,step_ids,case["gold_docs"],10)
        if step_count == len(step_results):
            a = generated
        else:
            a, _ = answer(llm,corrected_history,corrected_question,ret,step_ids[:5])
            offline_probe_calls += 1
        checkpoint[f"checkpoint_answer_em_step{step_count}"] = em(a["answer"],case["gold_answer"])
        checkpoint[f"checkpoint_answer_f1_step{step_count}"] = token_f1(a["answer"],case["gold_answer"])
        checkpoint[f"checkpoint_answer_subspan_em_step{step_count}"] = subspan_em(a["answer"],gold_subspans,case["gold_answer"])
        stages.update(checkpoint)
        checkpoints.append({"step":step_count, "top5":step_ids[:5], "probe_answer":a["answer"], "scores":checkpoint})
    initial_predicted = router["router_raw"]["tools"]
    valid_predicted = list(dict.fromkeys(x for x in initial_predicted if x.startswith("search_") and x.removeprefix("search_") in DOMAINS))
    gold_tools = case["gold_tools"]
    cited = generated["valid_evidence_doc_ids"]
    total_sec = prep["seconds"] + router["seconds"] + sum(s["seconds"] for s in step_results) + extra_llm_sec + gen_cost["seconds"]
    row = {
        "case_id": case["id"], "domain": case["domain"],
        "condition":input_condition, "strategy":strategy,"agent":agent,
        "architecture":architecture,"typo_aware":typo_aware,
        "typos_inserted":len(edits),
        "router_em": float(set(valid_predicted) == set(gold_tools) and len(valid_predicted) == len(set(valid_predicted))),
        "router_tool_f1":set_f1(valid_predicted,gold_tools),
        "router_selected_n":len(initial_predicted),
        "answer_em":em(generated["answer"],case["gold_answer"]),
        "answer_token_f1":token_f1(generated["answer"],case["gold_answer"]),
        "answer_subspan_em":subspan_em(generated["answer"],gold_subspans,case["gold_answer"]),
        "citation_doc_em":float(set(cited) == set(case["gold_docs"])),
        "citation_doc_f1":set_f1(cited,case["gold_docs"]),
        "hallucinated_citations_n":len(generated["hallucinated_citations"]),
        "search_steps":len(step_results),
        "offline_probe_calls":offline_probe_calls,
        "llm_calls":prep["calls"]+router["calls"]+extra_llm_calls+gen_cost["calls"],
        "llm_tokens":prep["tokens"]+router["tokens"]+extra_tokens+gen_cost["tokens"],
        "latency_s":round(total_sec,4),
        "preprocess_latency_s":round(prep["seconds"],4),
        "route_latency_s":round(router["seconds"],4),
        "retrieval_latency_s":round(sum(s["seconds"] for s in step_results),4),
        "reasoning_latency_s":round(extra_llm_sec,4),
        "generation_latency_s":round(gen_cost["seconds"],4),
        **stages}
    detail = {"case_id":case["id"], "condition":input_condition,"strategy":strategy,"agent":agent,
              "architecture":architecture,"typo_aware":typo_aware,
              "edits":edits, "original_history":case["history"], "original_question":case["question"],
              "variant_history":history,"variant_question":question,
              "corrected_history":corrected_history,"corrected_question":corrected_question,
              "gold_tools":gold_tools,"gold_docs":case["gold_docs"],"gold_spans":case["gold_spans"],
              "gold_answer":case["gold_answer"], "predicted_tools":initial_predicted,
              "router_query":query,"selection_and_search_query":search_query,
              "search_history":searched,"react_decisions":intermediate,
              "search_results":[{k:v for k,v in s.items() if k != "searches"} for s in step_results],
              "checkpoint_probes":checkpoints,
              "pred_answer":generated,"final_chunk_ids":final_ids[:10],"metrics":row}
    return row,detail


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if not ys: return float('nan')
    n = (len(ys)-1)*p
    low = int(n)
    return ys[low]+(ys[min(low+1,len(ys)-1)]-ys[low])*(n-low)


def summarize(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    retrieval_only = not all("answer_em" in r for r in rows)
    metrics = ["router_em","router_tool_f1","bm25_doc_recall@5","dense_doc_recall@5",
               "rrf_doc_recall@5","ce_doc_recall@5","ce_span_recall@5",
               "bm25_mrr@10","dense_mrr@10","rrf_mrr@10","ce_mrr@10",
                "final_doc_recall@5","final_span_recall@5","final_all_spans@5",
               "final_subspan_em@5","final_all_subspan_em@5",
                "answer_em","answer_subspan_em","answer_token_f1","citation_doc_em","citation_doc_f1",
                "llm_calls","llm_tokens","search_steps","latency_s", "preprocess_latency_s",
                "route_latency_s", "retrieval_latency_s","reasoning_latency_s","generation_latency_s"]
    metrics = [m for m in metrics if all(m in r and r[m] is not None for r in rows)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"],r["strategy"],r["agent"],r.get("architecture","staged"),r.get("typo_aware",False))].append(r)
    summary = []
    for (condition,strategy,agent,architecture,typo_aware),batch in sorted(groups.items()):
        s = {"condition":condition,"strategy":strategy,"agent":agent,
             "architecture":architecture,"typo_aware":typo_aware,"n":len(batch)}
        s.update({m: round(statistics.fmean(float(r[m]) for r in batch),4) for m in metrics})
        s["latency_p50_s"] = round(percentile([float(r["latency_s"]) for r in batch],.5),4)
        s["latency_p95_s"] = round(percentile([float(r["latency_s"]) for r in batch],.95),4)
        summary.append(s)
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)
    # Per-search-step traces; n_reached and reach_rate prevent conditional sample-size confusion.
    step_summary = []
    for (condition,strategy,agent,architecture,typo_aware),batch in sorted(groups.items()):
        for step in (1,2,3):
            part = [r for r in batch if r["search_steps"] >= step]
            if not part: continue
            entry = {"condition":condition,"strategy":strategy,"agent":agent,
                     "architecture":architecture,"typo_aware":typo_aware,
                     "step":step,"n_reached":len(part),"reach_rate":round(len(part)/len(batch),4)}
            entry["doc_recall@5"] = round(statistics.fmean(r[f"checkpoint_doc_recall@5_step{step}"] for r in part),4) if f"checkpoint_doc_recall@5_step{step}" in part[0] else float('nan')
            entry["span_recall@5"] = round(statistics.fmean(r[f"checkpoint_span_recall@5_step{step}"] for r in part),4) if f"checkpoint_span_recall@5_step{step}" in part[0] else float('nan')
            if f"checkpoint_answer_em_step{step}" in part[0]:
                entry["answer_em"] = round(statistics.fmean(r[f"checkpoint_answer_em_step{step}"] for r in part),4)
                entry["answer_subspan_em"] = round(statistics.fmean(r[f"checkpoint_answer_subspan_em_step{step}"] for r in part),4)
                entry["answer_token_f1"] = round(statistics.fmean(r[f"checkpoint_answer_f1_step{step}"] for r in part),4)
            if f"checkpoint_mrr@10_step{step}" in part[0]:
                entry["mrr@10"] = round(statistics.fmean(r[f"checkpoint_mrr@10_step{step}"] for r in part),4)
            keys=("condition","strategy","agent","architecture","typo_aware","step","n_reached","reach_rate","doc_recall@5","span_recall@5","mrr@10","answer_em","answer_subspan_em","answer_token_f1")
            step_summary.append({k:entry[k] for k in keys if k in entry})
    with (out/"step_summary.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(step_summary[0]));wr.writeheader();wr.writerows(step_summary)

    # Paired bootstrap deltas, same case IDs (never an independent two-sample test).
    rng = random.Random(48711)
    comparisons=[]
    if all("answer_em" in r for r in rows):
        for condition in ("clean","typo"):
            for agent in ("direct","react"):
              for architecture in sorted({r.get("architecture","staged") for r in rows}):
               for typo_aware in sorted({r.get("typo_aware",False) for r in rows}):
                for strategy,baseline in (("last","raw"),("full","raw"),("full","last")):
                    baseline_rows={r["case_id"]:r for r in rows if r["condition"]==condition and r["agent"]==agent and r["strategy"]==baseline and r.get("architecture","staged")==architecture and r.get("typo_aware",False)==typo_aware}
                    treatment={r["case_id"]:r for r in rows if r["condition"]==condition and r["agent"]==agent and r["strategy"]==strategy and r.get("architecture","staged")==architecture and r.get("typo_aware",False)==typo_aware}
                    ids=sorted(set(baseline_rows)&set(treatment))
                    for metric in ("router_em","ce_doc_recall@5","final_doc_recall@5","final_subspan_em@5","answer_em","answer_token_f1","answer_subspan_em","latency_s"):
                        deltas=[float(treatment[i][metric])-float(baseline_rows[i][metric]) for i in ids]
                        boot=[statistics.fmean(rng.choices(deltas,k=len(deltas))) for _ in range(2000)] if deltas else []
                        comparisons.append({"condition":condition,"agent":agent,"architecture":architecture,"typo_aware":typo_aware,"comparison":f"{strategy}_minus_{baseline}","metric":metric,"n":len(ids),
                                            "mean_delta":round(statistics.fmean(deltas),4) if deltas else None,
                                            "ci95_low":round(percentile(boot,.025),4) if boot else None,
                                            "ci95_high":round(percentile(boot,.975),4) if boot else None})
    # Matched ReAct - Direct effect for each typo × correction combination.
    if all("answer_em" in r for r in rows):
      for condition in ("clean","typo"):
        for strategy in ("raw","last","full"):
         for architecture in sorted({r.get("architecture","staged") for r in rows}):
          for typo_aware in sorted({r.get("typo_aware",False) for r in rows}):
            direct={r["case_id"]:r for r in rows if r["condition"]==condition and r["strategy"]==strategy and r["agent"]=="direct" and r.get("architecture","staged")==architecture and r.get("typo_aware",False)==typo_aware}
            react={r["case_id"]:r for r in rows if r["condition"]==condition and r["strategy"]==strategy and r["agent"]=="react" and r.get("architecture","staged")==architecture and r.get("typo_aware",False)==typo_aware}
            ids=sorted(set(direct)&set(react))
            for metric in ("final_doc_recall@5","final_subspan_em@5","answer_em","answer_token_f1","answer_subspan_em","latency_s","llm_calls"):
                deltas=[float(react[i][metric])-float(direct[i][metric]) for i in ids]
                boot=[statistics.fmean(rng.choices(deltas,k=len(deltas))) for _ in range(2000)] if deltas else []
                comparisons.append({"condition":condition,"agent":"react_vs_direct","architecture":architecture,"typo_aware":typo_aware,"comparison":f"react_minus_direct__{strategy}",
                                    "metric":metric,"n":len(ids),
                                    "mean_delta":round(statistics.fmean(deltas),4) if deltas else None,
                                    "ci95_low":round(percentile(boot,.025),4) if boot else None,
                                    "ci95_high":round(percentile(boot,.975),4) if boot else None})
    with (out/"paired_deltas.csv").open("w",newline="",encoding="utf-8") as f:
        comparison_fields=list(comparisons[0]) if comparisons else ["condition","agent","architecture","typo_aware","comparison","metric","n","mean_delta","ci95_low","ci95_high"]
        wr=csv.DictWriter(f,fieldnames=comparison_fields);wr.writeheader();wr.writerows(comparisons)
    report = (["# Measured retrieval-only benchmark results", "", "Answer generation and ReAct follow-up were skipped; this report contains routing and retrieval metrics only.", "",
                "| Condition | Correction | Architecture | Typo-aware | Agent | Router EM | CE Doc Recall@5 | CE Span Recall@5 | CE MRR@10 |",
                "|---|---|---|---|---|---:|---:|---:|---:|"] if retrieval_only and not all(r.get("router_skipped",False) for r in rows) else
              ["# Measured no-router retrieval diagnostic", "", "Router, query rewrite, answer generation, and ReAct were skipped. The gold domain index was used directly; routing metrics are not applicable.", "",
                "| Condition | Correction | Architecture | Typo-aware | Agent | BM25 Doc Recall@5 | Dense Doc Recall@5 | BM25 MRR@10 | Dense MRR@10 |",
                "|---|---|---|---|---|---:|---:|---:|---:|"] if retrieval_only else
              ["# Measured benchmark results", "", "These metrics are generated from actual row-level execution.", "",
                "| Condition | Correction | Architecture | Typo-aware | Agent | Router EM | CE Doc Recall@5 | Final Doc Recall@5 | Retrieval Subspan EM@5 | Answer EM | Answer Subspan EM | Answer Token F1 | Avg latency (s) |",
                "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in summary:
        if retrieval_only and all(r.get("router_skipped",False) for r in rows):
            report.append("| "+" | ".join(str(r[k]) for k in ("condition","strategy","architecture","typo_aware","agent","bm25_doc_recall@5","dense_doc_recall@5","bm25_mrr@10","dense_mrr@10"))+" |")
        elif retrieval_only:
            report.append("| "+" | ".join(str(r[k]) for k in ("condition","strategy","architecture","typo_aware","agent","router_em","ce_doc_recall@5","ce_span_recall@5","ce_mrr@10"))+" |")
        elif "answer_em" in r:
            report.append("| "+" | ".join(str(r[k]) for k in ("condition","strategy","architecture","typo_aware","agent","router_em","ce_doc_recall@5","final_doc_recall@5","final_subspan_em@5","answer_em","answer_subspan_em","answer_token_f1","latency_s"))+" |")
    if not retrieval_only:
        report += ["", "## ReAct checkpoint metrics (only cases that reached the step)", "",
                    "| Condition | Correction | Architecture | Typo-aware | Agent | Search step | Cases reached | Answer EM | Answer Subspan EM | Doc Recall@5 |",
                    "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
        for r in step_summary:
            if "answer_em" not in r:
                continue
            report.append("| "+" | ".join(str(r[k]) for k in ("condition","strategy","architecture","typo_aware","agent","step","n_reached","answer_em","answer_subspan_em","doc_recall@5"))+" |")
    (out/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("\nCOMPLETED: n =", len(rows))
    cols = ["condition","strategy","agent","router_em","ce_doc_recall@5","final_doc_recall@5","answer_em","answer_subspan_em","latency_s"]
    print("\t".join(cols))
    for row in summary:
        print("\t".join(str(row.get(c,"")) for c in cols))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default=os.getenv("BENCH_MODEL"),help="API-accessible Responses model ID; REQUIRED")
    p.add_argument("--data-zip",default=None,help="Path to official multidoc2dial.zip if already downloaded")
    p.add_argument("--out",default="./results")
    p.add_argument("--seed",type=int,default=20261020)
    p.add_argument("--n-per-domain",type=int,default=25)
    p.add_argument("--dense-model",default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    p.add_argument("--cross-model",default="cross-encoder/ms-marco-MiniLM-L6-v2")
    p.add_argument("--device",default="cpu")
    p.add_argument("--batch-size",type=int,default=64,
                   help="Batch size for offline dense-index embedding; smaller batches can reduce memory pressure")
    p.add_argument("--llm-concurrency",type=int,default=1,
                   help="Maximum simultaneous independent case evaluations")
    p.add_argument("--typo-types",default="transpose",
                   help="Comma-separated synthetic typo types: transpose,delete,insert,replace")
    p.add_argument("--typo-edits-per-word",type=int,default=1,
                   help="Number of sequential synthetic edits applied to each selected word in typo input")
    p.add_argument("--prepare-only",action="store_true",help="Download & select cases, without API/model inference")
    p.add_argument("--limit",type=int,default=None,help="Smoke-test a few cases; omit for full 100")
    p.add_argument("--architecture",choices=("staged","direct","both"),default="both",
                   help="staged=correction then route; direct=joint tool selection and query writing")
    p.add_argument("--typo-awareness",choices=("off","on","both"),default="both",
                   help="Whether router prompts explicitly account for typos")
    p.add_argument("--fresh-run",action="store_true",
                   help="Start a separate run directory and do not reuse preexisting rows")
    p.add_argument("--agent",choices=("direct","react","both"),default="both",
                   help="Run direct/no-loop, ReAct, or both")
    p.add_argument("--strategy",choices=("raw","last","full","all"),default="all",
                   help="Correction arm(s) to include")
    p.add_argument("--retrieval-only",action="store_true",
                   help="Stop after first retrieval; skip final answer generation and ReAct decisions")
    p.add_argument("--skip-router",action="store_true",
                   help="Skip LLM tool routing and retrieve directly from each case's gold domain (retrieval diagnostic only)")
    p.add_argument("--rewrite-only-search",action="store_true",
                   help="Use exactly one rewritten sentence as the search query; router selects tools only")
    p.add_argument("--retrieval-only-summary",action="store_true",
                   help="Summarize an existing retrieval-only JSONL run without calling the API")
    a=p.parse_args()
    typo_types=tuple(dict.fromkeys(x.strip() for x in a.typo_types.split(",") if x.strip()))
    valid_typo_types={"transpose","delete","insert","replace"}
    if not typo_types or set(typo_types)-valid_typo_types:
        raise SystemExit(f"--typo-types must be comma-separated values from {sorted(valid_typo_types)}")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    docs,dials=get_data(out/"data",a.data_zip)
    cases=build_cases(docs,dials,a.seed,a.n_per_domain)
    if a.limit: cases=cases[:a.limit]
    dump_json(out/"selected_cases.json",cases)
    architectures=["staged","direct"] if a.architecture=="both" else [a.architecture]
    typo_awareness=[False,True] if a.typo_awareness=="both" else [a.typo_awareness=="on"]
    agents=["direct","react"] if a.agent=="both" else [a.agent]
    strategies=["raw","last","full"] if a.strategy=="all" else [a.strategy]
    comparisons_per_case=2*len(strategies)*len(architectures)*len(typo_awareness)*len(agents)
    config_path=out/("run_config_fresh.json" if a.fresh_run else "run_config.json")
    dump_json(config_path,{**vars(a),"dataset":DATASET_URL,"case_ids":[x["id"] for x in cases],
                                             "architectures":architectures,"typo_aware_conditions":typo_awareness,
                                             "comparisons_per_case":comparisons_per_case,
                                             "total_target_evaluations":comparisons_per_case*len(cases)})
    print(f"Selected {len(cases)} cases; domains: {dict(Counter(c['domain'] for c in cases))}")
    if a.retrieval_only_summary:
        manifest=out/("rows_fresh.jsonl" if a.fresh_run else "rows.jsonl")
        if not manifest.exists():
            raise SystemExit(f"No row manifest to summarize: {manifest}")
        rows=[json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected={(c["id"],condition,strategy,agent,architecture,typo_aware)
                  for c in cases for condition in ("clean","typo") for strategy in strategies
                  for agent in agents for architecture in architectures for typo_aware in typo_awareness}
        keyed={(r["case_id"],r["condition"],r["strategy"],r["agent"],r.get("architecture","staged"),r.get("typo_aware",False)):r for r in rows}
        missing=expected-set(keyed)
        if missing:
            raise SystemExit(f"Cannot summarize incomplete run; missing {len(missing)} expected rows")
        all_rows=[keyed[k] for k in sorted(expected)]
        with (out/"rows_fresh.csv").open("w",newline="",encoding="utf-8") as f:
            fields=list(dict.fromkeys(k for row in all_rows for k in row))
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
        summarize(all_rows,out/"fresh_summary")
        return
    if a.prepare_only:
        return
    if not os.environ.get("OPENAI_API_KEY") or not a.model:
        raise SystemExit("Set OPENAI_API_KEY and BENCH_MODEL=your API-accessible model ID; or use --prepare-only")
    if a.fresh_run and not a.model.endswith("-luna"):
        raise SystemExit("Fresh final run must record the explicitly requested gpt-6-luna model ID")
    if a.retrieval_only and (a.agent != "direct" or a.strategy == "all" or a.architecture == "both"):
        raise SystemExit("--retrieval-only requires a single correction strategy, one architecture, and --agent direct")
    if a.skip_router and (not a.retrieval_only or a.strategy != "raw"):
        raise SystemExit("--skip-router is a retrieval-only diagnostic and requires --strategy raw")
    cache_name="model_cache_fresh.jsonl" if a.fresh_run else "model_cache.jsonl"
    llm=JsonModel(a.model,out/cache_name)
    chunks=build_chunks(docs)
    ret=Retrieval(chunks,a.dense_model,a.cross_model,a.device,docs,batch_size=a.batch_size)
    print(f"Built {len(chunks)} domain-indexed document chunks")
    manifest=out/("rows_fresh.jsonl" if a.fresh_run else "rows.jsonl")
    details=out/("details_fresh.jsonl" if a.fresh_run else "details.jsonl")
    existing={}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row=json.loads(line)
                existing[(row["case_id"],row["condition"],row["strategy"],row["agent"],
                          row.get("architecture","staged"),row.get("typo_aware",False))]=row
    jobs = [(case, condition, strategy, agent, architecture, typo_aware) for case in cases
            for condition in ("clean","typo") for strategy in strategies
            for agent in agents for architecture in architectures
            for typo_aware in typo_awareness]
    random.Random(a.seed + 91817).shuffle(jobs)  # avoid fixed arm-order/warmup bias
    pending=[]
    for case,condition,strategy,agent,architecture,typo_aware in jobs:
        key=(case["id"],condition,strategy,agent,architecture,typo_aware)
        if key not in existing:
            pending.append((key,case,condition,strategy,agent,architecture,typo_aware))

    def run_job(job):
        key,case,condition,strategy,agent,architecture,typo_aware=job
        result=evaluate_one(case,condition,strategy,agent,llm,ret,a.seed,architecture,typo_aware,
                            retrieval_only=a.retrieval_only,typo_types=typo_types,
                            skip_router=a.skip_router,edits_per_word=a.typo_edits_per_word,
                            rewrite_only_search=a.rewrite_only_search)
        return key,result

    def save_result(key,result):
        row,detail=result
        with manifest.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
        with details.open("a",encoding="utf-8") as f:
            f.write(json.dumps(detail,ensure_ascii=False)+"\n")
        existing[key]=row
        progress=("router skipped" if row.get("router_skipped",False)
                  else f"route EM={row['router_em']:.0f}")
        if "answer_em" in row:
            progress+=f" answer EM={row['answer_em']:.0f} subspan EM={row['answer_subspan_em']:.2f}"
        print(f"[{len(existing)}/{comparisons_per_case*len(cases)}] {key} {progress}",flush=True)

    concurrency=max(1,a.llm_concurrency)
    if concurrency==1:
        for job in pending:
            key,result=run_job(job)
            save_result(key,result)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures=[pool.submit(run_job,job) for job in pending]
            for future in as_completed(futures):
                key,result=future.result()
                save_result(key,result)
    all_rows=[existing[(c["id"],condition,strategy,agent,architecture,typo_aware)] for c in cases
              for condition in ("clean","typo") for strategy in strategies for agent in agents
              for architecture in architectures for typo_aware in typo_awareness]
    rows_csv=out/("rows_fresh.csv" if a.fresh_run else "rows.csv")
    with rows_csv.open("w",newline="",encoding="utf-8") as f:
        fieldnames=list(dict.fromkeys(k for row in all_rows for k in row))
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(all_rows)
    if a.fresh_run:
        # Direct results go to a separate subdirectory so stale legacy results
        # and unrelated partial configurations cannot be mixed in summaries.
        summarize(all_rows,out/"fresh_summary")
    else:
        summarize(all_rows,out)


if __name__=="__main__":
    main()
