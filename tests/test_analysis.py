import csv
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
import numpy as np
from compare_results import compare, KEYS
from diagnose import imbalance
from io_utils import save


class AnalysisTests(unittest.TestCase):
    def test_comparison_rejects_misaligned_origins_and_wrong_aggregate(self):
        # Ignored fixture directory; no real user data or TEST outcomes.
        root=Path(__file__).resolve().parents[1]/'data/test_tmp'/uuid.uuid4().hex
        left,right=root/'left',root/'right'
        for folder,offset in ((left,0.),(right,.1)):
            folder.mkdir(parents=True)
            scores=[.2+offset,.4+offset]
            save(folder/'COMPLETE.json',dict(split='test',origins=2,eligible_origins=2,candidates=4,
                                            **{k:float(np.mean(scores)) for k in KEYS}))
            with (folder/'SESSIONS.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['origin','impression','user',*KEYS]);writer.writeheader()
                for i,value in enumerate(scores):writer.writerow(dict(origin=i,impression=10+i,user=20+i,**{k:value for k in KEYS}))
        result=compare(left,right)
        self.assertEqual(result['matched_origins'],2)
        self.assertAlmostEqual(result['delta_left_minus_right']['ndcg10'],-.1)
        file=right/'SESSIONS.csv';text=file.read_text(encoding='utf-8')
        file.write_text(text.replace('0,10,20,','0,999,20,'),encoding='utf-8')
        with self.assertRaisesRegex(AssertionError,'impression'):compare(left,right)
        file.write_text(text,encoding='utf-8')
        save(right/'COMPLETE.json',dict(split='test',origins=2,eligible_origins=2,candidates=4,**{k:.9 for k in KEYS}))
        with self.assertRaisesRegex(AssertionError,'aggregate'):compare(left,right)

    def test_imbalance_counts_no_click_and_all_click_without_test_labels(self):
        data=SimpleNamespace(base_info={'splits':{'train':{'origins':3}}},
            a={'offsets':np.array([0,2,4,6,8]),'train_labels':np.array([1,0,0,0,1,1])},
            fit=np.array([0,1]),valid=np.array([2]),lengths=np.array([2,2,2,2]))
        result=imbalance(data)
        self.assertEqual(result['FIT']['no_click_origins'],1)
        self.assertEqual(result['FIT']['ranking_eligible_origins'],1)
        self.assertEqual(result['FIT']['candidate_click_fraction'],.25)
        self.assertEqual(result['VALID']['all_click_origins'],1)


if __name__=='__main__':unittest.main()
