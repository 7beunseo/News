import html
import numpy as np
from .config import ROOT,SEEDS,RUN_ORDER,LABELS
from io_utils import read,save,now


def report():
    rows=[];results=[]
    for seed in SEEDS:
        for mode in RUN_ORDER:
            run=ROOT/'runs'/str(seed)/mode
            val=read(run/'SELECTED_ON_VALID.json') if (run/'SELECTED_ON_VALID.json').exists() else {}
            test=read(run/'research_TEST/COMPLETE.json') if (run/'research_TEST/COMPLETE.json').exists() else {}
            progress=read(run/'PROGRESS.json') if (run/'PROGRESS.json').exists() else {'state':'대기'}
            cells=[str(seed),LABELS[mode],progress['state'],str(val.get('epoch','—'))]
            cells += [f'{test[k]:.6f}' if k in test else '미완료' for k in ('auc','mrr','ndcg5','ndcg10')]
            rows.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
            results.append(dict(seed=seed,mode=mode,selected=val,test=test,progress=progress))
    summaries={}
    for mode in RUN_ORDER:
        done=[r['test'] for r in results if r['mode']==mode and r['test']]
        if done:summaries[mode]=dict(completed_seeds=len(done),**{k:dict(mean=float(np.mean([v[k] for v in done])),
            sd=float(np.std([v[k] for v in done],ddof=1)) if len(done)>1 else None) for k in ('auc','mrr','ndcg5','ndcg10')})
    queue=read(ROOT/'PROGRESS.json') if (ROOT/'PROGRESS.json').exists() else {'state':'준비 중'}
    text='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>HCCN 진행 및 TEST 결과</title>
<style>body{font:16px/1.7 sans-serif;max-width:1150px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}th{background:#eef3fa}</style>
<h1>HCCN 진행 및 TEST 결과</h1>'''
    text+=f'<p>갱신 {now()} · 큐 {html.escape(queue["state"])}</p>'
    text+='<p>MIND-small, 6조건 × 3seed = 18회 학습 예정, 조건당 5에폭. 완료 수와 예정 수는 다릅니다. 모든 설정을 먼저 동결하고, 동일 seed의 모든 모델에서 VALID 최고를 선택한 후 연구용 TEST를 평가합니다.</p>'
    text+='<table><tr><th>seed</th><th>모델</th><th>진행</th><th>선택 epoch</th><th>TEST AUC</th><th>TEST MRR</th><th>TEST NDCG@5</th><th>TEST NDCG@10</th></tr>'+''.join(rows)+'</table>'
    text+='<p>핵심 비교는 A4 Full vs A3 공통 관심사입니다. A5는 rho만 0으로 만들므로 모든 관계 효과를 제거하지 않습니다. 각 모델의 인코더·활성 파라미터·계산량은 완전히 같지 않습니다. 같은 seed의 A1–A5 초기 전체 파라미터와 샘플 순서는 일치합니다.</p>'
    text+='<p>공식 DEV 전체를 재사용한 연구용 TEST입니다. 독립 미사용 TEST나 공식 challenge TEST가 아닙니다. 미완료 점수는 VALID로 대체하지 않습니다. 단일 seed 순위만으로 통계적 우위·신규성을 확정하지 않습니다.</p></html>'
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'RESULT_KO.html').write_text(text,encoding='utf-8')
    save(ROOT/'RESULT.json',dict(time=now(),queue=queue,runs=results,summaries=summaries))


if __name__=='__main__':report()
