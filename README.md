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

최대 5에폭 중 VALID NDCG@10 최고 에폭을 선택하고 동률이면 이른 에폭을 사용합니다.
학습 중 완료된 에폭만으로 별도 TEST를 요청하려면 `evaluate_best.py`에
`--interim`을 명시합니다. TEST 결과를 체크포인트 선택에 사용하지 않습니다.
중간 TEST를 반복 관찰한 경우 미사용 독립 검증이라고 표현해서는 안 됩니다.
선택된 에폭 번호는 모델마다 달라도 됩니다. 비교할 학습 예산을 동일하게 맞춘 뒤
각 모델에서 VALID 최고를 선택합니다. 저장 기록에 6에폭 이후 결과가 있더라도
이 평가 도구는 최대 5에폭 범위 안에서만 선택하며 TEST 점수로 재선택하지 않습니다.

## 전체 비교 큐

```bash
python -u run_queue.py --dataset small
```

기본 큐는 데이터셋당 4조건 × 1seed = 4회 학습입니다. Small과 Large를 모두 실행하면
총 8회 학습이며, 8개 독립 seed가 아닙니다. 다중 seed 재현은 별도 후속 확인입니다.
같은 seed의 모든 조건을 학습한 뒤 VALID 선택을 동결하고 TEST를 평가합니다.
공통 인코더 초기값과 노출 순서의 일치를 검사합니다.

| 순서 | 조건 | 학습 손실 |
|---|---|---|
| 1 | `nrms_bce` | 전체 후보 BCE |
| 2 | `nrms_bce_pair` | BCE + 0.2 Pair |
| 3 | `rcrin_no_exchange` | BCE + 0.2 Pair + 0.25 Role + 0.2 Edge, 역할 교환 제거 |
| 4 | `rcrin_full` | 동일 손실, 양방향 역할 교환 적용 |

Pair는 같은 노출에서 클릭한 뉴스의 점수가 비클릭 뉴스보다 높아지도록 하는
`softplus(-(s_positive - s_negative))`의 쌍별 평균입니다.
1↔2는 Pair, 2↔3은 보조 Role/Edge 감독, 3↔4는 교환 연산의 효과를 비교합니다.
1↔3에는 Pair도 달라지므로 Role/Edge만의 효과라고 해석하지 않습니다.
모델 파라미터 수와 연산량이 완전히 같은 비교는 아닙니다.

원래 NRMS의 sampled-softmax CE 학습과 위 전체 후보 BCE 학습은 다른 조건입니다.
다른 학습법·seed·배치 크기의 기존 실행을 비교할 때는 구조만의 인과 효과로 해석하지 않습니다.

## NRMS Sampled CE + Pair 비교

기존의 강한 Sampled CE 조건에 클릭·비클릭 Pairwise Loss만 추가하는 별도 실험입니다.
위의 **전체 후보 BCE + Pair와 다른 학습 조건**이며 관계 모듈은 추가하지 않습니다.

```bash
python -u sampled_ce.py --dataset small --mode ce --seed 26111701
python -u sampled_ce.py --dataset small --mode ce_pair --seed 26111701
```

먼저 위의 `prepare.py`로 데이터를 준비합니다. 이 두 실행은 각 실행 폴더에
설정·소스·데이터 해시를 별도 동결하므로 기존 네 조건의 비교 큐를 변경하지 않습니다.
각 명령은 **5에폭 학습 → VALID 최고 선택 → 전체 연구용 TEST 평가**를 수행합니다.
`large`도 같은 명령으로 지원하지만, 코드를 지원한다는 것이 해당 실험 완료를 뜻하지는 않습니다.

두 조건은 같은 NRMS, 무작위 초기값, seed, batch 64, microbatch 4, Adam 0.001,
학습 사례 순서와 비클릭 샘플을 사용합니다. 하나의 노출에 클릭이 여러 개면
클릭별로 학습 사례를 만들고, 그 노출의 비클릭 4개를 뽑습니다.
비클릭이 4개보다 적을 때만 복원 추출합니다. TEST 정답은 학습·샘플링에 사용하지 않습니다.

```text
CE:       L = CrossEntropy([s_click, s_nonclick_1, ..., s_nonclick_4], target=0)
CE+Pair:  L = CE + 0.2 * mean_j softplus(s_nonclick_j - s_click)
```

두 손실은 **동일한 네 비클릭**에 적용합니다. Pair 계수 0.2는 기존 Pair 대조군과
같은 고정값이며 이 실행에서 탐색하지 않습니다. `pair_loss.py`는 현재 별도 작업
폴더에서 실행 중인 실험과 동일한 손실 코드입니다. `sampled_ce.py`는 그 학습 규칙을
이 저장소의 전처리·평가 코드에서 실행하도록 옮긴 배포본이며, 외부 작업 폴더를
필요로 하지 않습니다. 별도의 대규모 학습을 완료했다는 의미는 아닙니다.

출력은 `runs/<dataset>/<seed>/nrms_sampled_ce[_pair]/`에 저장됩니다.
`EPOCHS.json`에는 에폭별 VALID 점수와 샘플 해시가, `research_TEST/`에는 선택한
모델의 TEST 점수와 노출별 지표가 기록됩니다. 두 조건이 모두 완료되면 초기값,
5에폭 샘플 순서, 소스, 입력 데이터와 TEST 노출을 검산하고
`comparisons/<dataset>_<seed>_sampled_ce_pair/`에 비교표를 자동 생성합니다.
단일 seed에서의 차이는 통계적 우위나 독립 재현을 의미하지 않습니다.
이전에 반복 관찰한 공식 DEV는 미사용 TEST라고 표현하지 않습니다.

## VALID/FIT 진단

```bash
python -u diagnose.py --dataset small --mode rcrin_full --seed 26111801 --out diagnostics/small_rcrin
```

완료된 1~5에폭 중 VALID 최고 체크포인트를 고정하고 같은 가중치에서 관계 보정만
제거한 점수, 역할 확률, 방향별 유효 attention 질량, gate, 클릭 불균형을 기록합니다.
기본적으로 고정 FIT 최대 16개에서 각 완료 에폭의 공유 뉴스 인코더 기울기도 측정합니다.
TEST 정답은 열지 않으며 optimizer 갱신은 없습니다. 출력 보정 제거는 재학습 ablation과 다르고,
소수 FIT 표본의 gradient cosine은 전체 학습의 인과관계를 입증하지 않습니다.
진단에는 CUDA가 필요하며 `--skip-gradients`로 기울기 진단을 생략할 수 있습니다.

## 저장된 TEST 결과 비교

```bash
python compare_results.py --left runs/small/26111801/nrms_bce/research_TEST --right runs/small/26111801/nrms_bce_pair/research_TEST --left-name NRMS-BCE --right-name NRMS-BCE-Pair --out comparisons/small_bce_pair
```

두 폴더의 `COMPLETE.json`과 `SESSIONS.csv`를 읽어 평가 원점·노출·사용자 순서,
지표 평균을 검산하고 CSV/JSON을 저장합니다. 재학습이나 TEST 기반 선택을 하지 않습니다.
다른 epoch/seed/데이터 분할을 통제했는지는 별도로 확인해야 합니다.
단일 모델의 TEST가 아직 없다면 먼저 `evaluate_best.py`를 사용합니다.
생성되는 데이터·가중치·예측·진단 결과는 Git 추적 대상에서 제외됩니다.

세 모델 이상을 한 표에 넣을 때는 `--additional 이름 결과폴더`를 반복합니다.

```bash
python compare_results.py --left runs/small/26111801/nrms_bce/research_TEST --right runs/small/26111801/nrms_bce_pair/research_TEST --left-name NRMS-BCE --right-name NRMS-BCE-Pair --additional RCRIN runs/small/26111801/rcrin_full/research_TEST --out comparisons/small_three_models
```

이 경우 첫 모델을 기준으로 차이를 기록하고 입력한 순서를 유지합니다.
TEST 점수로 승자 모델이나 checkpoint를 다시 선택하는 기능은 없습니다.
별도 평가 도구를 사용했다면 해당 `manual_evaluations/.../research_TEST` 경로를 지정할 수 있습니다.

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
| `selection.py` | 최대 5에폭의 VALID 최고 선택, 동률·유효성 검사 |
| `diagnose.py` | VALID 보정 제거·역할/attention·FIT 기울기 진단 |
| `compare_results.py` | 동일 원점 TEST 결과의 검산과 비교표 |
| `sampled_ce.py`, `pair_loss.py` | 같은 샘플의 NRMS CE / CE+Pair 학습·VALID 선택·TEST 비교 |
| `tests/` | 모델·시간 인과성·지표 검사 |

NRMS는 [Wu et al. (2019)](https://aclanthology.org/D19-1671/)의 논문 기반
PyTorch 재구현입니다. 공식 TensorFlow 코드 그대로가 아니며, 무작위 word
초기화를 사용하고 GloVe를 사용하지 않습니다. 이 저장소는 standalone 배포본이며
경로·전처리·큐 관리를 분리했습니다. 과거 실행의 bitwise 재현을 보장하지 않습니다.
