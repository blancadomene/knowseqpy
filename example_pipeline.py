"""
Example pipeline using breast samples.
"""

import logging
import os
from datetime import datetime

from knowseqpy.counts_to_matrix import counts_to_matrix


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        filename=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_knowseq_logs.log",
        filemode="w"
    )

    # Set and read paths
    script_path = os.path.dirname(os.path.abspath(__file__))
    info_path = os.path.join(script_path, "tests", "test_fixtures", "samples_info_breast.csv")
    counts_path = os.path.join(script_path, "tests", "test_fixtures", "breast_count_files")

    # Execute counts to matrix conversion
    counts_matrix, labels = counts_to_matrix(info_path=info_path, counts_path=counts_path)

    # Number of samples per class
    print(labels.value_counts())


if __name__ == '__main__':
    main()
