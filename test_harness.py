"""Offline structural tests, using miniature SYNTHETIC data and a mock LLM.
These are NOT benchmark scores and must not be reported as such.
"""
import tempfile
import random
from pathlib import Path
import benchmark as b


class MockLLM:
    def ask(self, sys, prompt, schema):
        import json
        name = schema["name"]
        if name == "fix_last":
            result = {"text":json.loads(prompt)["last_user_utterance_to_fix"]}
        elif name == "fix_full":
            result = {"utterances":json.loads(prompt)["user_utterances_to_fix_in_order"]}
        elif name == "multi_tool_route":
            result = {"tools":["search_dmv"],"query":"what application documents are needed"}
        elif name == "rewrite_query":
            result = {"query":"what application documents are needed"}
        elif name == "react_decision":
            result = {"action":"search","tool":"search_dmv","query":"additional requirements details"}
        elif name == "answer_with_citations":
            result = {"answer":"You need documents.","evidence_doc_ids":["dmv_doc"]}
        else:
            raise AssertionError(name)
        return result, .15, 123


class MockRetrieval:
    def __init__(self, docs):
        ch=b.build_chunks(docs)
        self.chunks={x.id:x for x in ch}
        self.source_docs=docs
        self.first=ch[0].id

    def run_tools(self, tools, query):
        return {"searches":[],"bm25_ids":[self.first],"dense_ids":[self.first],
                "rrf_ids":[self.first],"ce_ids":[self.first],"top_ids":[self.first],"seconds":.05}

    def search_gold_domain(self, domain, query):
        return {"searches":[],"bm25_ids":[self.first],"dense_ids":[self.first],
                "rrf_ids":[self.first],"ce_ids":[self.first],"top_ids":[self.first],"seconds":.05}


def run():
    corrupted, edit = b.corrupt_one("application",random.Random(1),
                                   ("transpose","delete","insert","replace"))
    assert corrupted != "application"
    assert edit["type"] in {"transpose","delete","insert","replace"}
    more_corrupted, more_edit = b.corrupt_one("application",random.Random(1),
                                              ("transpose","delete","insert","replace"),3)
    assert more_corrupted != "application"
    assert len(more_edit["operations"])==3
    docs={domain:{} for domain in b.DOMAINS}
    dialogs={domain:[] for domain in b.DOMAINS}
    for domain in b.DOMAINS:
        did=f'{domain}_doc'
        docs[domain][did]={"title":"Application Process", "doc_text":"Documents are required for the application process.",
                            "spans":{"1":{"start_sp":0,"end_sp":22}}}
        dialogs[domain].append({"dial_id":f'{domain}_conversation',"turns":[
            {"role":"user","turn_id":1,"utterance":"Can I apply?"},
            {"role":"agent","turn_id":2,"utterance":"Yes, you can.","references":[]},
            {"role":"user","turn_id":3,"utterance":"What documents do I need?"},
            {"role":"agent","turn_id":4,"utterance":"You need documents.",
             "references":[{"label":"solution","id_sp":"1","doc_id":did}]}
        ]})
    cases=b.build_cases(docs,dialogs,9,n_per_domain=1)
    assert len(cases)==4
    c=next(c for c in cases if c['domain']=='dmv')
    llm=MockLLM();ret=MockRetrieval(docs)
    for condition in ['clean','typo']:
        r,d=b.evaluate_one(c,condition,'raw','direct',llm,ret,seed=9,
                           architecture='staged',typo_aware=True,retrieval_only=True,
                           refine_router_input=True)
        assert r['router_em']==1.0
        assert r['llm_calls']==2
        assert d['selection_and_search_query']==d['router_query']
        assert 'answer_em' not in r
        r,d=b.evaluate_one(c,condition,'raw','direct',llm,ret,seed=9,
                           architecture='staged',typo_aware=True,retrieval_only=True,
                           rewrite_only_search=True)
        assert r['router_em']==1.0
        assert r['rewrite_only_search'] is True
        assert r['llm_calls']==2
        assert d['selection_and_search_query']==d['router_query']
        assert 'answer_em' not in r
        r,d=b.evaluate_one(c,condition,'raw','direct',llm,ret,seed=9,
                           architecture='direct',typo_aware=False,retrieval_only=True,
                           skip_router=True)
        assert r['router_skipped'] is True
        assert r['router_em'] is None
        assert r['llm_calls']==0
        assert 'answer_em' not in r
        assert d['predicted_tools'] is None
        r,d=b.evaluate_one(c,condition,'raw','direct',llm,ret,seed=9,
                           architecture='direct',typo_aware=True,retrieval_only=True)
        assert r['router_em']==1.0
        assert r['ce_doc_recall@5']==1.0
        assert 'answer_em' not in r
        assert 'pred_answer' not in d
        assert r['llm_calls']==1
        r,d=b.evaluate_one(c,condition,'last','direct',llm,ret,seed=9,
                           architecture='staged',typo_aware=True,retrieval_only=True)
        assert r['router_em']==1.0
        assert r['ce_doc_recall@5']==1.0
        assert 'answer_em' not in r
        assert r['llm_calls']==3
    rows=[]
    for condition in ['clean','typo']:
        for strategy in ['raw','last','full']:
            for agent in ['direct','react']:
                for arch in ['direct','staged']:
                    r,d=b.evaluate_one(c,condition,strategy,agent,llm,ret,seed=9,
                                       architecture=arch,typo_aware=True)
                    assert r['answer_em']==1.0
                    assert 0.0 <= r['answer_subspan_em'] <= 1.0
                    assert r['router_em']==1.0
                    assert r['ce_doc_recall@5']==1.0
                    assert r['checkpoint_answer_em_step1']==1.0
                    assert 0.0 <= r['checkpoint_answer_subspan_em_step1'] <= 1.0
                    assert r['search_steps']==(3 if agent=='react' else 1)
                    if agent=='react':
                        assert r['checkpoint_answer_em_step3']==1.0
                    rows.append(r)
    with tempfile.TemporaryDirectory() as tmp:
        b.summarize(rows,Path(tmp))
        assert (Path(tmp)/'summary.csv').exists()
        assert (Path(tmp)/'paired_deltas.csv').exists()
    print('PASS - synthetic benchmark harness structural checks (not measured results)')

if __name__=='__main__': run()
