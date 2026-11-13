import types
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from nrms import NRMS
from pair_loss import components, summed_loss
from sampled_ce import SampledNRMS, initial_hash, positive_examples, sample_candidates


class SampledCEChecks(unittest.TestCase):
    def test_zero_pair_matches_original_ce_and_gradient(self):
        scores = torch.tensor([[.8, -.4, .7, -1.2, 0.], [-.3, 1.2, 0., .5, -.1]], requires_grad=True)
        expected = F.cross_entropy(scores, torch.zeros(2, dtype=torch.long), reduction='sum')/2
        actual = summed_loss(scores, 0., 2)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(torch.autograd.grad(expected, scores, retain_graph=True)[0],
                                   torch.autograd.grad(actual, scores)[0]))

    def test_pair_gradient_and_microbatch_scaling(self):
        scores = torch.tensor([[.8, -.4, .7, -1.2, 0.], [-.3, 1.2, 0., .5, -.1]], requires_grad=True)
        _, pair = components(scores)
        manual = torch.stack([torch.stack([F.softplus(scores[i,j]-scores[i,0]) for j in range(1,5)]).mean() for i in range(2)])
        torch.testing.assert_close(pair, manual)
        gradient = torch.autograd.grad(pair.sum(), scores, retain_graph=True)[0]
        self.assertTrue((gradient[:,0] < 0).all())
        self.assertTrue((gradient[:,1:] > 0).all())
        full = summed_loss(scores, .2, 2)
        split = sum(summed_loss(scores[i:i+1], .2, 2) for i in range(2))
        torch.testing.assert_close(torch.autograd.grad(full, scores, retain_graph=True)[0],
                                   torch.autograd.grad(split, scores)[0])

    def test_positive_pool_and_same_impression_sampling(self):
        # Last impression has no stored labels: TEST must never enter the pool.
        data = types.SimpleNamespace(fit=np.array([1,0]), a=dict(
            offsets=np.array([0,3,8,10]), items=np.arange(10,20),
            train_labels=np.array([1,1,0,0,0,1,0,0])))
        positives = positive_examples(data)
        np.testing.assert_array_equal(positives[0], [0,0,1])
        np.testing.assert_array_equal(positives[1], [10,11,15])
        chosen = np.array([2,0,1])
        qq, cc = sample_candidates(data, positives, chosen, np.random.default_rng(41))
        np.testing.assert_array_equal(qq, [1,0,0])
        np.testing.assert_array_equal(cc[:,0], [15,10,11])
        self.assertEqual(set(cc[0,1:]), {13,14,16,17})
        np.testing.assert_array_equal(cc[1:,1:], np.full((2,4), 12))
        qq2, cc2 = sample_candidates(data, positives, chosen, np.random.default_rng(41))
        np.testing.assert_array_equal(cc, cc2)

    def test_adapter_preserves_parameters_scores_and_cache(self):
        torch.manual_seed(26111701)
        baseline = NRMS(12).eval()
        torch.manual_seed(26111701)
        adapted = SampledNRMS(12, 'nrms_sampled_ce_pair').eval()
        self.assertEqual(initial_hash(baseline), initial_hash(adapted))
        titles = torch.tensor([[0,0,0], [1,2,0], [3,4,5], [6,7,0]])
        history = torch.tensor([[1,2], [0,0]])
        candidates = torch.tensor([[2,3], [1,2]])
        with torch.no_grad():
            expected = baseline(titles, history, candidates)
            actual = adapted(titles, history, candidates)['scores']
            cache = adapted.encode_news(titles)
            cached = adapted.cached(cache, history, candidates)['scores']
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        torch.testing.assert_close(actual, cached, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
