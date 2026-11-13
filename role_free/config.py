from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
from config import DATA_ROOT
CFG=dict(epochs=5,seed=26111701,lr=.001,gradient_clip=4,
    batch_size=32,microbatch_max=8,max_micro_pair_cells=65536,
    relation_dim=64,heads=4,rounds=2,pair_chunk=32,news_chunk=256,
    sampled_negatives=4,relation_weight=.2,pair_weight=.2,
    selection='max full VALID NDCG@10, earliest tie',early_stopping=False,
    training_unit='one impression; mean CE over clicked items; mean over impressions',
    full_impression=True,source_of_current_labels='loss only')
MODES=('rf_full','rf_no_relation_loss','plain_attention_pair','nrms_ce_pair','rf_aux_only','nrms_ce')
RUN_ORDER=('rf_full','rf_no_relation_loss','plain_attention_pair','nrms_ce_pair')
