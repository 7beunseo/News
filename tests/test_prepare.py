import json
from pathlib import Path
import uuid
import unittest
import numpy as np
from prepare import prepare

class PreparationTests(unittest.TestCase):
    def test_chronology_fit_only_history_and_hidden_dev_outcomes(self):
        scratch=Path(__file__).resolve().parents[1]/'data'/'test_tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        # Keep tiny fixtures under ignored data/ for debugging. Default ACLs work
        # in restricted Windows sessions where tempfile's mode=0700 may not.
        root=scratch/uuid.uuid4().hex
        assert root.resolve().is_relative_to(scratch.resolve())
        root.mkdir();train=root/'train';dev=root/'dev';train.mkdir();dev.mkdir()
        news=''.join(f'N{i}\tcategory\tsub\tNews title {i}\t\t\t\t\n' for i in [1,2,3])
        for folder in (train,dev):(folder/'news.tsv').write_text(news,encoding='utf-8')
        rows=[f'{i+1}\tU1\t11/{i+1:02d}/2019 10:00:00 AM\tN1\tN2-1 N3-0\n' for i in range(20)]
        (train/'behaviors.tsv').write_text(''.join(rows),encoding='utf-8')
        # A nonnumeric DEV outcome makes accidental label parsing fail.
        (dev/'behaviors.tsv').write_text('1\tU1\t11/25/2019 10:00:00 AM\tN1\tN2-HIDDEN N3-HIDDEN\n',encoding='utf-8')
        out=root/'prepared';prepare(train,dev,out,'small')
        info=json.loads((out/'COMPLETE.json').read_text());self.assertFalse(info['dev_labels_parsed'])
        self.assertEqual(info['fit_origins'],18);self.assertEqual(info['valid_origins'],2)
        self.assertEqual(len(np.load(out/'train_labels.npy')),40)
        history=np.load(out/'matured.npy');self.assertEqual(np.count_nonzero(history[-1]),19)
        self.assertEqual(np.count_nonzero(history[1]),1)  # Exactly 24 hours is excluded.
        self.assertEqual(np.count_nonzero(history[2]),2)
