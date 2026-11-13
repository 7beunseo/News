"""Standalone NRMS sampled-CE / sampled-CE+Pair controlled comparison.

Uses the repository's prepared data and NRMS, without importing private research
directories. Only the recommendation loss differs between the two conditions.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
import traceback

import numpy as np
import torch

from config import ROOT
from data import Impressions
from io_utils import now, read, save, sha
from nrms import NRMS
from pair_loss import components
from selection import select_validation_best
from train import evaluate, progress, seed_all

SETTINGS = dict(epochs=5, batch_size=64, microbatch=4, lr=.001,
                negative_ratio=4, gradient_clip=4, pair_weight=.2,
                early_stopping=False, selection='max VALID NDCG@10; earliest tie')


class SampledNRMS(NRMS):
    """Output adapter for the existing whole-TEST evaluator; no new parameters."""
    def __init__(self, vocab_size, mode):
        super().__init__(vocab_size)
        self.mode = mode

    def forward(self, titles, history, candidates, mask=None):
        return {'scores': super().forward(titles, history, candidates)}

    def cached(self, cache, history, candidates, mask=None):
        user = self.encode_user(cache[history], history.ne(0))
        return {'scores': (cache[candidates] * user[:, None, :]).sum(-1).float()}


def positive_examples(data):
    """One example per clicked item, in original impression / candidate order."""
    origins, items = [], []
    for q in np.sort(data.fit):
        lo, hi = map(int, data.a['offsets'][q:q+2])
        labels = data.a['train_labels'][lo:hi]
        assert len(labels) == hi-lo and set(np.unique(labels)) <= {0, 1}
        if not np.any(labels == 1):
            continue
        if not np.any(labels == 0):
            raise ValueError(f'FIT origin {q} has no non-clicked candidate.')
        clicked = data.a['items'][lo:hi][labels == 1]
        origins.extend([int(q)] * len(clicked))
        items.extend(map(int, clicked))
    if not items:
        raise ValueError('No eligible FIT positive examples.')
    return np.asarray(origins, np.int64), np.asarray(items, np.int64)


def sample_candidates(data, positives, chosen, rng):
    qq = positives[0][chosen]
    cc = np.empty((len(qq), 5), np.int64)
    cc[:, 0] = positives[1][chosen]
    for j, q in enumerate(qq):
        lo, hi = map(int, data.a['offsets'][q:q+2])
        negatives = data.a['items'][lo:hi][data.a['train_labels'][lo:hi] == 0]
        assert len(negatives) > 0
        cc[j, 1:] = rng.choice(negatives, 4, replace=len(negatives) < 4)
    return qq, cc


def initial_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def optimizer_step(model, data, titles, qq, cc, optimizer, pair_weight):
    optimizer.zero_grad(set_to_none=True)
    totals = np.zeros(2)
    for st in range(0, len(qq), SETTINGS['microbatch']):
        q = qq[st:st + SETTINGS['microbatch']]
        h = torch.tensor(np.array(data.a['history'][q], dtype=np.int64), device=titles.device)
        c = torch.tensor(cc[st:st + SETTINGS['microbatch']], device=titles.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            scores = model(titles, h, c)['scores']
        ce, pair = components(scores)
        loss = (ce.sum() + pair_weight * pair.sum()) / len(qq)
        assert torch.isfinite(loss)
        loss.backward()
        totals += [float(ce.detach().sum()), float(pair.detach().sum())]
    torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return totals


def compare_if_complete(dataset, seed):
    parent = ROOT / 'runs' / dataset / str(seed)
    paths = [parent / name for name in ('nrms_sampled_ce', 'nrms_sampled_ce_pair')]
    if not all((p / 'research_TEST/COMPLETE.json').exists() for p in paths):
        return
    starts = [read(p / 'STARTED.json') for p in paths]
    records = [read(p / 'EPOCHS.json') for p in paths]
    assert starts[0]['initial_sha256'] == starts[1]['initial_sha256']
    protocols = [read(p / 'RUN_PROTOCOL.json') for p in paths]
    assert protocols[0]['data_sha256'] == protocols[1]['data_sha256']
    assert protocols[0]['source_sha256'] == protocols[1]['source_sha256']
    assert protocols[0]['settings'] == protocols[1]['settings']
    assert [r['epoch'] for r in records[0]] == [r['epoch'] for r in records[1]] == list(range(1, 6))
    assert [r['sampling_sha256'] for r in records[0]] == [r['sampling_sha256'] for r in records[1]]
    from compare_results import compare
    result = compare(paths[1] / 'research_TEST', paths[0] / 'research_TEST',
                     'NRMS Sampled CE + Pair', 'NRMS Sampled CE')
    result.update(training_conditions_not_checked=False, same_initial_weights=True,
                  same_five_epoch_sample_streams=True, same_prepared_data_and_source=True,
                  independently_trained_seeds=1)
    destination = ROOT / 'comparisons' / f'{dataset}_{seed}_sampled_ce_pair'
    if (destination / 'COMPARISON.json').exists():
        return
    save(destination / 'COMPARISON.json', result)
    with (destination / 'COMPARISON.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'selected_epoch', 'auc', 'mrr', 'ndcg5', 'ndcg10'])
        writer.writeheader()
        for i, side in enumerate(('right', 'left')):
            writer.writerow(dict(model=result[side + '_name'],
                selected_epoch=select_validation_best(records[i])['epoch'],
                **{k: result[side][k] for k in ('auc', 'mrr', 'ndcg5', 'ndcg10')}))


def run(dataset, condition, seed):
    mode = 'nrms_sampled_ce' + ('_pair' if condition == 'ce_pair' else '')
    pair_weight = SETTINGS['pair_weight'] if condition == 'ce_pair' else 0.
    output = ROOT / 'runs' / dataset / str(seed) / mode
    if (output / 'research_TEST/COMPLETE.json').exists():
        compare_if_complete(dataset, seed)
        print('Already complete; existing results preserved.', flush=True)
        return
    data = Impressions(dataset)
    if not (output / 'TRAINING_COMPLETE.json').exists():
        if (output / 'RUN_PROTOCOL.json').exists() or (output / 'STARTED.json').exists():
            raise RuntimeError('Existing incomplete run preserved; use a separate checkout to restart.')
        assert SETTINGS['epochs'] == 5
        source_names = ('sampled_ce.py', 'pair_loss.py', 'nrms.py', 'data.py', 'config.py',
                        'train.py', 'metrics.py', 'io_utils.py', 'selection.py', 'compare_results.py')
        data_names = ('COMPLETE.json', 'titles.npy', 'matured.npy', 'train_origins.npy',
                      'valid_origins.npy', 'items.npy', 'offsets.npy', 'train_labels.npy')
        save(output / 'RUN_PROTOCOL.json', dict(time=now(), dataset=dataset, mode=mode,
            seed=seed, settings=SETTINGS, effective_pair_weight=pair_weight,
            source_sha256={n: sha(ROOT/n) for n in source_names},
            data_sha256={n: sha(data.prep/n) for n in data_names}, test_used_for_selection=False))
        positives = positive_examples(data)
        seed_all(seed)  # Seed once; preserve the legacy dropout stream across epochs.
        titles = torch.tensor(np.array(data.a['titles'], dtype=np.int64), device='cuda')
        model = SampledNRMS(data.info['vocab_size'], mode).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['lr'])
        save(output / 'STARTED.json', dict(time=now(), seed=seed, initial_sha256=initial_hash(model),
            parameters=sum(p.numel() for p in model.parameters()), positive_examples=len(positives[0]),
            previous_weights_loaded=False, dataset=dataset, mode=mode, effective_pair_weight=pair_weight))
        records = []
        for epoch in range(1, 6):
            model.train()
            rng = np.random.default_rng(seed + epoch)
            order = rng.permutation(len(positives[0]))
            digest = hashlib.sha256()
            totals = np.zeros(2)
            seen = 0
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for batch, st in enumerate(range(0, len(order), SETTINGS['batch_size'])):
                qq, cc = sample_candidates(data, positives, order[st:st+SETTINGS['batch_size']], rng)
                digest.update(qq.tobytes())
                digest.update(cc.tobytes())
                totals += optimizer_step(model, data, titles, qq, cc, optimizer, pair_weight)
                seen += len(qq)
                if batch % 100 == 0:
                    progress(output, state='TRAINING', model=mode, dataset=dataset, seed=seed,
                             epoch=epoch, epochs=5, batch=batch, batches=math.ceil(len(order)/64),
                             ce=totals[0]/seen, pair=totals[1]/seen)
                    if (ROOT/'STOP_REQUEST.json').exists():
                        raise RuntimeError('USER_STOP_REQUEST: existing results preserved')
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            memory = torch.cuda.max_memory_allocated()/2**20
            checkpoint = output / f'epoch{epoch:02d}.pt'
            torch.save(dict(model=model.state_dict(), epoch=epoch, mode=mode, seed=seed,
                            settings=SETTINGS, effective_pair_weight=pair_weight), checkpoint)
            val = evaluate(model, data, titles, 'valid', output/f'valid_epoch{epoch:02d}', output)
            entry = dict(epoch=epoch, validation_ndcg10=val['ndcg10'],
                         training_loss=(totals[0]+pair_weight*totals[1])/seen,
                         ce=totals[0]/seen, pair=totals[1]/seen, train_seconds=seconds,
                         peak_cuda_MiB=memory, sampling_sha256=digest.hexdigest(),
                         checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
            records.append(entry)
            save(output/'EPOCHS.json', records)
            save(output/'SELECTED_ON_VALID.json', select_validation_best(records))
            print(json.dumps(entry), flush=True)
        selected = select_validation_best(records)
        save(output/'TRAINING_COMPLETE.json', dict(time=now(), selected=selected,
             fits=1, seed=seed, epochs=5, test_used_for_selection=False))
    protocol = read(output/'RUN_PROTOCOL.json')
    for name, expected in protocol['source_sha256'].items():
        assert sha(ROOT/name) == expected, f'Source changed after freeze: {name}'
    for name, expected in protocol['data_sha256'].items():
        assert sha(data.prep/name) == expected, f'Data changed after freeze: {name}'
    selected = select_validation_best(read(output/'EPOCHS.json'))
    assert sha(selected['checkpoint']) == selected['checkpoint_sha256']
    frozen = output/'SELECTED_FROZEN.json'
    if not frozen.exists():
        save(frozen, dict(time=now(), dataset=dataset, mode=mode, seed=seed,
                         models={mode: selected}, test_used_for_selection=False))
    assert read(frozen)['models'][mode] == selected
    seed_all(seed)
    model = SampledNRMS(data.info['vocab_size'], mode).cuda()
    model.load_state_dict(torch.load(selected['checkpoint'], map_location='cuda', weights_only=False)['model'])
    model.requires_grad_(False)
    titles = torch.tensor(np.array(data.a['titles'], dtype=np.int64), device='cuda')
    result = evaluate(model, data, titles, 'test', output/'research_TEST', output, frozen)
    progress(output, state='COMPLETE', model=mode, dataset=dataset, seed=seed, selected_epoch=selected['epoch'])
    print(json.dumps(result), flush=True)
    compare_if_complete(dataset, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['small', 'large'], required=True)
    parser.add_argument('--mode', choices=['ce', 'ce_pair'], required=True)
    parser.add_argument('--seed', type=int, default=26111701)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        run(args.dataset, args.mode, args.seed)
    except Exception:
        mode = 'nrms_sampled_ce' + ('_pair' if args.mode == 'ce_pair' else '')
        save(ROOT/'runs'/args.dataset/str(args.seed)/mode/'ERROR.json',
             dict(time=now(), traceback=traceback.format_exc()))
        raise
