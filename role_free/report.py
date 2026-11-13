import html
from .config import ROOT,RUN_ORDER
from io_utils import read,now,save

NAMES={'rf_full':'제안: 관계 감독 + 관계 Attention','rf_no_relation_loss':'관계 감독 제거',
    'plain_attention_pair':'일반 후보 Attention + 점수 Pair','nrms_ce_pair':'NRMS + 점수 Pair'}


def report():
    rows=[];json_rows=[]
    for mode in RUN_ORDER:
        run=ROOT/'runs'/mode;selected=read(run/'SELECTED_ON_VALID.json') if (run/'SELECTED_ON_VALID.json').exists() else {}
        result=read(run/'research_TEST/COMPLETE.json') if (run/'research_TEST/COMPLETE.json').exists() else {}
        progress=read(run/'PROGRESS.json') if (run/'PROGRESS.json').exists() else {}
        values=[NAMES[mode],progress.get('state','대기'),str(selected.get('epoch','—'))]
        values += [f'{result[k]:.6f}' if k in result else '미완료' for k in ('auc','mrr','ndcg5','ndcg10')]
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
        json_rows.append(dict(mode=mode,selected=selected,test=result,progress=progress))
    queue=read(ROOT/'PROGRESS.json') if (ROOT/'PROGRESS.json').exists() else {'state':'준비 중'}
    text='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Role-Free 관계 학습 진행 및 결과</title>
<style>body{max-width:1100px;margin:35px auto;font:16px/1.7 sans-serif;padding:20px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccc}th{background:#eee}code{background:#eee}</style>
<h1>Role-Free 관계 학습: 진행 및 결과</h1>'''
    text+=f'<p>갱신: {html.escape(now())} · 큐 상태: <b>{html.escape(queue["state"])}</b></p>'
    text+='<p><a href="README.md">구조·가설·학습 조건·선행연구 설명</a></p>'
    text+='<p>MIND-small만 사용합니다. 조건별 5에폭, 하나의 seed입니다. 내부 VALID NDCG@10으로 선택한 체크포인트를 모든 모델에서 동결한 후 공식 DEV 전체를 연구용 TEST로 평가합니다. 연구용 TEST는 이미 과거 실험에서 사용되어 독립 최종 검증이 아닙니다.</p>'
    text+='<table><tr><th>모델</th><th>진행</th><th>선택 epoch</th><th>TEST AUC</th><th>TEST MRR</th><th>TEST NDCG@5</th><th>TEST NDCG@10</th></tr>'+''.join(rows)+'</table>'
    text+='<p>빈 점수는 아직 미실행/미완료입니다. VALID 점수를 TEST로 대체하지 않습니다. 학습 4조건은 독립 seed 4회가 아닙니다. rf_aux_only 및 nrms_ce는 구현·검산만 하였으며 자동 학습 큐에는 포함하지 않았습니다.</p>'
    text+='<p>새 통제군은 노출 단위 평균 CE를 동일하게 사용합니다. 기존 양성 사례 단위 CE 실험과 목적함수 가중 단위가 달라, 과거 TEST와의 차이를 구조만의 효과라고 해석할 수 없습니다. 각 조건의 계산량은 같지 않으며 시간·메모리를 함께 저장합니다.</p>'
    text+='<p>관계 감독 제거 실험은 감독의 효과, 일반 Attention 대조군은 관계 벡터를 사용하는 설계의 추가 효과를 검사합니다. 관계 보조 학습만으로 이득이 생겼는지까지 완전히 분리하려면 rf_aux_only 후속 학습이 필요합니다.</p></html>'
    (ROOT/'RESULT_KO.html').write_text(text,encoding='utf-8')
    save(ROOT/'RESULT.json',dict(time=now(),queue=queue,models=json_rows))


if __name__=='__main__':report()
