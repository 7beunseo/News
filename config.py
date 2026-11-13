from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
DATA_ROOT=Path(os.environ.get('NEWS_DATA_ROOT',str(ROOT/'data'/'prepared'))).resolve()
CFG=dict(epochs=8,batch_size=32,microbatch_max=16,max_micro_pair_cells=262144,
         lr=.001,optimizer='Adam',gradient_clip=4,dropout=.2,history=50,title_tokens=32,
         relation_dim=64,heads=4,rounds=2,pair_chunk=64,query_chunk=64,news_chunk=128,
         pair_checkpoint=True,role_weight=.25,edge_weight=.20,pair_weight=.20,
         selection='max full internal VALID NDCG@10; earliest epoch on tie',
         full_impression=True,negative_sampling=False,candidate_order_features=False,
         primary_metric='ndcg10',early_stopping=False)
SEEDS=(26111801,26111802,26111803)
# Starts with the requested proposal, then its fair baselines and ablations.
RUN_ORDER=('rcrin_full','nrms_bce','nrms_bce_pair','rcrin_unanchored',
           'rcrin_p_from_n','rcrin_n_from_p','rcrin_no_interaction',
           'rcrin_state_concat','rcrin_no_exchange')
DATASETS=('small','large')
