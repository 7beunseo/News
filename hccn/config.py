from pathlib import Path
import os
from config import DATA_ROOT
CODE=Path(__file__).resolve().parent
REPO=CODE.parent
ROOT=Path(os.environ.get('HCCN_OUTPUT_ROOT',str(CODE))).resolve()
PREPARED=Path(os.environ.get('HCCN_PREPARED_DIR',str(DATA_ROOT/'small'))).resolve()
CFG=dict(epochs=5,batch_size=64,microbatch_max=2,max_micro_pair_cells=32768,
    lr=.001,gradient_clip=4,dropout=.2,word_dim=300,dim=256,news_layers=2,
    heads=4,feedforward=512,slots=4,relation_dim=64,relation_hidden=128,
    pair_chunk=16,news_chunk=64,title_tokens=32,history=50,negatives=4,
    loss='sampled CE only; mean over clicked examples',
    selection='max full VALID NDCG@10 in epochs 1-5; earliest tie',early_stopping=False)
SEEDS=(26111701,26111702,26111703)
RUN_ORDER=('hccn_full','common_interest','individual','self_attention','no_rho','nrms_ce')
LABELS=dict(hccn_full='A4 HCCN Full',common_interest='A3 Common interest graph',
    individual='A1 Multi-interest individual',self_attention='A2 Candidate Self-Attention',
    no_rho='A5 Zero rho',nrms_ce='A0 NRMS CE')
