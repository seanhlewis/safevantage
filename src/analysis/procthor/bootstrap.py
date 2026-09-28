#!/usr/bin/env python3
# Paired scene-bootstrap for the fresh cohort. Stdlib only.
# Delta = counterevidence_active MINUS baseline, for macro_f1/risk/answer_rate/travel.
import json, gzip, random, sys, os

# Supply the evaluation root and output path through environment variables.
# SAFEvantage data files are not shipped in this public code package.
DATA_ROOT = os.environ.get("SAFEVANTAGE_DATA_ROOT", "data/derived")
EVAL_TPL = os.environ.get("SV_EVAL_TPL", DATA_ROOT + "/eval_budget{b}")
CONFIRM_OUT = os.environ.get("SV_CONFIRM", "results/fresh_cohort_bootstrap.json")
REF = "counterevidence_active"
BASELINES = os.environ.get("SV_BASELINES","max_observability,detector_confidence_gain,decision_resolve").replace(":",",").split(",")
NB = 10000
random.seed(0)

def load(budget):
    p = EVAL_TPL.format(b=budget)+"/rows_all.jsonl"
    op = (gzip.open(p+".gz","rt") if os.path.exists(p+".gz") else open(p))
    rows=[json.loads(l) for l in op]
    return rows

def f1(tp,fp,fn):
    if tp==0: return 0.0
    p=tp/(tp+fp) if (tp+fp) else 0.0; r=tp/(tp+fn) if (tp+fn) else 0.0
    return 0.0 if (p+r)==0 else 2*p*r/(p+r)

def counts_for(rows):
    # per (scene, method) sufficient stats
    C={}
    scenes=set()
    for x in rows:
        s=x["scene_id"]; m=x["method"]; scenes.add(s)
        d=C.setdefault((s,m),dict(tp_p=0,fp_p=0,fn_p=0,tp_n=0,fp_n=0,fn_n=0,risk=0.0,trav=0.0,ans=0,n=0))
        pred=x["prediction"]; truth=bool(x["truth"])
        # positive class = present/YES ; negative class = absent/NO ; wait = neither
        if pred=="yes" and truth: d["tp_p"]+=1
        if pred=="yes" and not truth: d["fp_p"]+=1
        if pred=="no" and not truth: d["tp_n"]+=1
        if pred=="no" and truth: d["fp_n"]+=1
        if truth and pred!="yes": d["fn_p"]+=1
        if (not truth) and pred!="no": d["fn_n"]+=1
        d["risk"]+=x["risk"]; d["trav"]+=x["geodesic_m"]
        d["ans"]+= (1 if pred in ("yes","no") else 0); d["n"]+=1
    return C, sorted(scenes)

def agg(C, scenes, m):
    t=dict(tp_p=0,fp_p=0,fn_p=0,tp_n=0,fp_n=0,fn_n=0,risk=0.0,trav=0.0,ans=0,n=0)
    for s in scenes:
        d=C.get((s,m))
        if not d: continue
        for k in t: t[k]+=d[k]
    macro=0.5*(f1(t["tp_p"],t["fp_p"],t["fn_p"])+f1(t["tp_n"],t["fp_n"],t["fn_n"]))
    n=max(t["n"],1)
    return dict(macro_f1=macro, risk=t["risk"]/n, answer_rate=t["ans"]/n, travel=t["trav"]/n)

def pctl(v,q):
    v=sorted(v); i=q*(len(v)-1); lo=int(i); hi=min(lo+1,len(v)-1); fr=i-lo
    return v[lo]*(1-fr)+v[hi]*fr

def run(budget, summary_check):
    rows=load(budget); C,scenes=counts_for(rows)
    full={m:agg(C,scenes,m) for m in [REF]+BASELINES}
    # sanity vs evaluator summary
    chk={}
    if summary_check and os.path.exists(summary_check):
        sm=json.load(open(summary_check)).get("methods",{})
        for m in full:
            if m in sm:
                chk[m]=dict(f1_ours=round(full[m]["macro_f1"],4), f1_eval=round(sm[m].get("macro_f1",-1),4),
                            risk_ours=round(full[m]["risk"],4), risk_eval=round(sm[m].get("mean_risk",-1),4))
    out=dict(budget=budget, n_scenes=len(scenes), n_episodes_per_method=len(rows)//(len(BASELINES)+1),
             point={m:full[m] for m in full}, sanity=chk, deltas={})
    for b in BASELINES:
        d={k:[] for k in ("macro_f1","risk","answer_rate","travel")}
        for _ in range(NB):
            samp=[random.choice(scenes) for _ in scenes]
            a=agg(C,samp,REF); bb=agg(C,samp,b)
            for k in d: d[k].append(a[k]-bb[k])
        res={}
        for k in d:
            pt=full[REF][k]-full[b][k]; lo=pctl(d[k],0.025); hi=pctl(d[k],0.975)
            fav=sum(1 for x in d[k] if (x<0 if k in("risk","travel") else x>0))/NB
            res[k]=dict(delta=round(pt,4), lo=round(lo,4), hi=round(hi,4), p_favorable=round(fav,4))
        out["deltas"][b]=res
    return out

if __name__=="__main__":
    result={}
    for bud in (8,12):
        sc=EVAL_TPL.format(b=bud)+"/summary_all.json"
        result[str(bud)]=run(bud, sc)
    outp=CONFIRM_OUT
    json.dump(result, open(outp,"w"), indent=2)
    print("WROTE", outp)
    for bud in ("8","12"):
        r=result[bud]; print(f"\n=== budget {bud}  scenes={r['n_scenes']}  eps/method={r['n_episodes_per_method']} ===")
        print("SANITY (ours vs evaluator):", r["sanity"].get(REF))
        pt=r["point"]
        print(f"  {REF}: f1={pt[REF]['macro_f1']:.4f} risk={pt[REF]['risk']:.4f} ans={pt[REF]['answer_rate']:.4f} travel={pt[REF]['travel']:.2f}")
        for b in BASELINES:
            print(f"  vs {b}: dF1={r['deltas'][b]['macro_f1']['delta']:+.4f} CI[{r['deltas'][b]['macro_f1']['lo']:+.4f},{r['deltas'][b]['macro_f1']['hi']:+.4f}] p={r['deltas'][b]['macro_f1']['p_favorable']:.3f} | dRisk={r['deltas'][b]['risk']['delta']:+.4f} CI[{r['deltas'][b]['risk']['lo']:+.4f},{r['deltas'][b]['risk']['hi']:+.4f}]")
