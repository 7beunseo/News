import unittest
import numpy as np
from metrics import metric,independent_metrics

class MetricTests(unittest.TestCase):
    def test_ties_multiple_clicks_and_random_scores(self):
        rng=np.random.default_rng(451)
        for n in [2,5,20,80]:
            for _ in range(15):
                y=rng.integers(0,2,n);y[0]=0;y[-1]=1
                for scores in [np.zeros(n),rng.normal(size=n),rng.integers(0,3,n)]:
                    a,b=metric(y,scores),independent_metrics(y,scores)
                    for key in a:self.assertAlmostEqual(a[key],b[key],places=12)
        self.assertIsNone(metric(np.array([0,0]),np.array([1,2])))
        self.assertIsNone(metric(np.array([1,1]),np.array([1,2])))
