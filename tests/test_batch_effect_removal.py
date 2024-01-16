import os
import unittest

import pandas as pd

from knowseq.batch_effect_removal import batch_effect_removal


class BatchEffectRemovalTest(unittest.TestCase):
    def setUp(self):
        batch_path = os.path.normpath(os.path.join("test_fixtures", "golden", "batch_matrix_sva_breast.csv"))
        self.golden_batch = pd.read_csv(batch_path, index_col=0, header=0)

    def test_rna_seq_qa(self):
        qa_matrix_path = os.path.normpath(
            os.path.join("test_fixtures", "golden", "qa_matrix_breast.csv"))
        qa_matrix = pd.read_csv(qa_matrix_path, header=0, index_col=0)

        res_qa, _ = batch_effect_removal(qa_matrix, "sva")
        pd.testing.assert_frame_equal(res_qa, self.golden_batch, check_dtype=False, check_like=True,
                                      check_exact=False, atol=0.1, rtol=0.1)

    if __name__ == '__main__':
        unittest.main()
