"""HCCN: full-context choice graph, positive-example Sampled CE only."""
import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from .config import ROOT, CODE, REPO, PREPARED, CFG, RUN_ORDER, SEEDS
from .data import Impressions
from .model import HCCN
from nrms import NRMS
from io_utils import read, save, sha, now
from .evaluation import evaluate, progress


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


class CEData(Impressions):
    def __init__(self):
        super().__init__('small')
        origins=[];items=[]
        for q in np.sort(self.fit):
            lo,hi=map(int,self.a['offsets'][q:q+2]);y=self.a['train_labels'][lo:hi]
            clicked=self.a['items'][lo:hi][y>0]
            if len(clicked) and not (y==0).any():raise ValueError('FIT positive has no negative')
            origins.extend([int(q)]*len(clicked));items.extend(map(int,clicked))
        self.a['positive_origins']=np.asarray(origins,np.int64)
        self.a['positive_items']=np.asarray(items,np.int64)
        self.a['fit_mask']=np.zeros(len(self.a['history']),bool);self.a['fit_mask'][self.fit]=True


class NRMSControl(NRMS):
    def __init__(self,vocab):super().__init__(vocab,dropout=CFG['dropout']);self.mode='nrms_ce'
    def forward(self,titles,h,c,mask):
        # Same full candidate encoding/microbatch protocol as the HCCN arms.
        b,nh=h.shape;nc=c.shape[1];ids=torch.cat([h,c.masked_fill(~mask,0)],1).flatten();keep=ids.ne(0)
        tokens=titles[ids[keep]]
        e=torch.cat([self.encode_news(tokens[i:i+CFG['news_chunk']]) for i in range(0,len(tokens),CFG['news_chunk'])],0)
        x=e.new_zeros(len(ids),256);x[keep]=e;x=x.reshape(b,nh+nc,256)
        user=self.encode_user(x[:,:nh],h.ne(0))
        return dict(scores=(x[:,nh:]*user[:,None,:]).sum(-1).float().masked_fill(~mask,0))
    def cached(self,cache,h,c,mask):
        u=self.encode_user(cache[h],h.ne(0))
        return dict(scores=(cache[c.masked_fill(~mask,0)]*u[:,None,:]).sum(-1).float().masked_fill(~mask,0))


def new_model(vocab,mode):
    if mode=='nrms_ce':return NRMSControl(vocab)
    keys=('word_dim','dim','news_layers','heads','feedforward','slots','relation_dim','relation_hidden','dropout','pair_chunk','news_chunk','title_tokens')
    return HCCN(vocab,mode,**{k:CFG[k] for k in keys})


def state_hash(model, backbone_only=False):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if backbone_only and isinstance(model,HCCN) and not name.startswith('news.'):
            continue
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def candidates_for(data, chosen, rng):
    a = data.a
    qq = a['positive_origins'][chosen]
    cc = np.empty((len(qq), 5), np.int64)
    cc[:, 0] = a['positive_items'][chosen]
    assert a['fit_mask'][qq].all()
    for j, q in enumerate(qq):
        lo, hi = map(int, a['offsets'][q:q+2])
        negatives = a['items'][lo:hi][a['train_labels'][lo:hi] == 0]
        assert len(negatives) > 0
        cc[j, 1:] = rng.choice(negatives, 4, replace=len(negatives) < 4)
    return qq, cc


def loss_columns(data, qq, sampled):
    """Labels are used outside forward, solely to locate the sampled loss terms."""
    columns = np.empty_like(sampled)
    for row, q in enumerate(qq):
        lo, hi = map(int, data.a['offsets'][q:q+2])
        items = data.a['items'][lo:hi]
        labels = data.a['train_labels'][lo:hi]
        for j, item in enumerate(sampled[row]):
            found = np.flatnonzero((items == item) & (labels == (1 if j == 0 else 0)))
            assert len(found) > 0, (int(q), int(item), j)
            columns[row, j] = found[0]
    return columns


def sampled_ce(scores, columns, full_batch_size):
    picked = scores.float().gather(1, columns)
    return F.cross_entropy(picked, torch.zeros(len(picked), device=scores.device,
                                               dtype=torch.long), reduction='sum') / full_batch_size


def step(model, data, titles, qq, sampled, optimizer):
    optimizer.zero_grad(set_to_none=True)
    columns = loss_columns(data, qq, sampled)
    offset = 0; value = 0.
    for micro in data.microbatches(qq):
        h, c, mask = data.inputs(micro, titles.device)
        ix = torch.tensor(columns[offset:offset+len(micro)], device=titles.device)
        expected = torch.tensor(sampled[offset:offset+len(micro)], device=titles.device)
        assert torch.equal(c.gather(1, ix), expected)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(titles, h, c, mask)
        loss = sampled_ce(output['scores'], ix, len(qq))
        assert torch.isfinite(loss)
        loss.backward(); value += float(loss.detach()); offset += len(micro)
    assert offset == len(qq)
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return value


def verify_frozen():
    frozen = read(ROOT/'FROZEN_PROTOCOL.json')
    assert frozen['config'] == CFG and frozen['seeds']==list(SEEDS)
    assert frozen['prepared_directory']==str(PREPARED)
    for path, expected in frozen['source_sha256'].items():
        assert sha(path) == expected, path
    for path, expected in frozen['data_sha256'].items():assert sha(path)==expected,path
    return frozen


def train(mode,seed):
    frozen = verify_frozen()
    run = ROOT/'runs'/str(seed)/mode
    if (run/'TRAINING_COMPLETE.json').exists(): return
    if (run/'STARTED.json').exists(): raise RuntimeError('Existing incomplete run preserved')
    seed_all(seed); data = CEData()
    titles = torch.tensor(np.array(data.a['titles'], dtype=np.int64), device='cuda')
    model = new_model(data.info['vocab_size'], mode).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    full_hash = state_hash(model); backbone_hash = state_hash(model, True)
    family='nrms_ce' if mode=='nrms_ce' else 'hccn_full'
    assert full_hash==frozen['initial_full_sha256'][str(seed)][family]
    save(run/'STARTED.json', dict(time=now(), pid=os.getpid(), mode=mode, seed=seed, config=CFG,
        initial_full_sha256=full_hash, initial_encoder_sha256=backbone_hash,
        parameters=sum(p.numel() for p in model.parameters()), previous_weights_loaded=False,
        positive_examples=len(data.a['positive_items']), training_loss='sampled CE only',
        context='whole impression; outcomes never enter forward'))
    records = []; start_all = time.perf_counter()
    for epoch in range(1, 6):
        model.train(); rng = np.random.default_rng(seed+epoch)
        order = rng.permutation(len(data.a['positive_items']))
        digest = hashlib.sha256(); total = 0.; seen = 0
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        for batch, st in enumerate(range(0, len(order), CFG['batch_size'])):
            qq, cc = candidates_for(data, order[st:st+CFG['batch_size']], rng)
            digest.update(np.asarray(qq, dtype=np.int64).tobytes()); digest.update(cc.tobytes())
            loss = step(model, data, titles, qq, cc, optimizer)
            total += loss*len(qq); seen += len(qq)
            if batch % 50 == 0:
                progress(run, state='TRAINING', mode=mode, epoch=epoch, epochs=5,
                    batch=batch, batches=math.ceil(len(order)/CFG['batch_size']),
                    ce_loss=total/seen, pair_weight=0, role_weight=0, edge_weight=0)
                if (ROOT/'STOP_REQUEST.json').exists(): raise RuntimeError('USER_STOP_REQUEST')
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started; memory = torch.cuda.max_memory_allocated()/2**20
        assert digest.hexdigest() == frozen['expected_sampling_sha256'][str(seed)][str(epoch)]
        checkpoint = run/f'epoch{epoch:02d}.pt'
        torch.save(dict(model=model.state_dict(), mode=mode, epoch=epoch, config=CFG), checkpoint)
        temporary = run/'last_optimizer_state.pt.tmp'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), mode=mode,
            epoch=epoch, config=CFG, cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temporary)
        os.replace(temporary, run/'last_optimizer_state.pt')
        val = evaluate(model, data, titles, 'valid', run/f'valid_epoch{epoch:02d}', run)
        entry = dict(epoch=epoch, training_ce=total/seen, train_seconds=seconds,
            peak_cuda_MiB=memory, validation_ndcg10=val['ndcg10'],
            sampling_sha256=digest.hexdigest(), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
        records.append(entry); save(run/'EPOCHS.json', records)
        selected = max(records, key=lambda r:(r['validation_ndcg10'], -r['epoch']))
        save(run/'SELECTED_ON_VALID.json', selected)
        print(json.dumps(entry), flush=True)
    save(run/'TRAINING_COMPLETE.json', dict(time=now(), selected=selected, config=CFG,
        seed=seed, fits=1, seconds=time.perf_counter()-start_all,
        initial_full_sha256=full_hash, initial_encoder_sha256=backbone_hash,
        sampling_hashes={str(r['epoch']):r['sampling_sha256'] for r in records},
        test_used_for_selection=False))


def test(mode,seed):
    verify_frozen()
    frozen_path = ROOT/'frozen'/f'{seed}.json'; frozen = read(frozen_path)
    assert set(frozen['models']) == set(RUN_ORDER) and not frozen['test_used_for_selection']
    entry = frozen['models'][mode]
    assert sha(entry['checkpoint']) == entry['checkpoint_sha256']
    seed_all(seed); data = CEData(); model = new_model(data.info['vocab_size'], mode).cuda()
    model.load_state_dict(torch.load(entry['checkpoint'], map_location='cuda', weights_only=False)['model'])
    model.requires_grad_(False)
    titles = torch.tensor(np.array(data.a['titles'], dtype=np.int64), device='cuda')
    run = ROOT/'runs'/str(seed)/mode
    from .diagnostics import diagnose
    diagnose(model,data,titles,run)
    result = evaluate(model, data, titles, 'test', run/'research_TEST', run, frozen_path)
    progress(run, state='TEST_COMPLETE', mode=mode, selected_epoch=entry['epoch'], **{k:result[k] for k in ('auc','mrr','ndcg5','ndcg10')})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode', choices=RUN_ORDER, required=True)
    p.add_argument('--test', action='store_true');p.add_argument('--seed',type=int,choices=SEEDS,required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    try: test(args.mode,args.seed) if args.test else train(args.mode,args.seed)
    except BaseException:
        save(ROOT/'runs'/str(args.seed)/args.mode/('TEST_ERROR.json' if args.test else 'ERROR.json'),dict(time=now(), traceback=traceback.format_exc()))
        raise
