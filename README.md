# News

RCRIN(Reciprocal Cross-Role Interaction Network)과 NRMS 대조군의 뉴스 추천 연구 코드입니다.
모델, 데이터 전처리, 학습, VALID 체크포인트 선택, TEST 평가 스크립트를 포함합니다.
데이터셋·가중치·개별 예측·실험 결과는 포함하지 않습니다.

## 설치 및 검사

Python 3.10 이상과 CUDA 호환 PyTorch 2.5를 사용합니다.
단위 검사는 CPU에서, 학습·전체 평가는 CUDA GPU에서 실행됩니다.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## 데이터 준비

[MIND 공식 사이트](https://msnews.github.io/)의 이용 조건을 확인하고
MIND-small 또는 MIND-large의 TRAIN/DEV를 다운로드하여 압축을 해제합니다.
각 폴더에는 `news.tsv`, `behaviors.tsv`가 필요합니다.

```bash
python prepare.py --dataset small --train-dir data/raw/small/train --dev-dir data/raw/small/dev
python freeze.py --datasets small
```

공식 TRAIN을 timestamp 순 약 90:10으로 FIT/VALID로 나누며 timestamp 동률은
같은 구간에 둡니다. 공식 DEV 전체를 연구용 TEST로 사용합니다.
정답이 비공개인 공식 challenge TEST의 평가와 구분해야 합니다.
어휘는 FIT에서 보이는 제목만으로 구성하고, 추가 이력은 24시간보다 오래된
FIT 클릭만 사용합니다. 현재 노출의 정답은 모델 forward 입력에 넣지 않습니다.

`large`도 동일 명령에서 데이터셋 이름과 원본 경로를 변경해 준비할 수 있습니다.
둘 다 준비한 뒤에는 `python freeze.py --datasets small large`로 함께 동결합니다.
전처리 저장 위치는 `NEWS_DATA_ROOT` 환경변수로 지정할 수 있습니다.
기본 위치는 `data/prepared`입니다. 기존 준비 데이터와 동결 파일은 덮어쓰지 않습니다.

## 단일 모델 학습과 평가

```bash
python -u train.py --dataset small --mode rcrin_full --seed 26111801
python -u evaluate_best.py --dataset small --mode rcrin_full --seed 26111801 --out manual_evaluations/small_best
```

8에폭 중 VALID NDCG@10 최고 에폭을 선택하고 동률이면 이른 에폭을 사용합니다.
학습 중 완료된 에폭만으로 별도 TEST를 요청하려면 `evaluate_best.py`에
`--interim`을 명시합니다. TEST 결과를 체크포인트 선택에 사용하지 않습니다.
중간 TEST를 반복 관찰한 경우 미사용 독립 검증이라고 표현해서는 안 됩니다.

## 전체 비교 큐

```bash
python -u run_queue.py --dataset small
```

기본 큐는 데이터셋당 9조건 × 3seed = 27회 학습입니다.
같은 seed의 모든 조건을 학습한 뒤 VALID 선택을 동결하고 TEST를 평가합니다.
공통 인코더 초기값과 노출 순서의 일치를 검사합니다.

## 파일

| 파일 | 역할 |
|---|---|
| `model.py` | RCRIN 역할 교환, 후보 관계, 보정 및 손실 |
| `nrms.py` | 공통 뉴스·사용자 인코더 |
| `prepare.py`, `data.py` | 시간순 분할과 전체 후보 입력 |
| `train.py` | 학습·VALID 선택·추론 |
| `metrics.py` | tie-aware 지표와 독립 구현 검산 |
| `freeze.py`, `run_queue.py` | 설정 동결과 비교 큐 |
| `evaluate_best.py` | 선택한 모델의 별도 TEST |
| `tests/` | 모델·시간 인과성·지표 검사 |

NRMS는 [Wu et al. (2019)](https://aclanthology.org/D19-1671/)의 논문 기반
PyTorch 재구현입니다. 공식 TensorFlow 코드 그대로가 아니며, 무작위 word
초기화를 사용하고 GloVe를 사용하지 않습니다. 이 저장소는 standalone 배포본이며
경로·전처리·큐 관리를 분리했습니다. 과거 실행의 bitwise 재현을 보장하지 않습니다.
