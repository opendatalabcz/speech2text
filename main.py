import sys
import os

from speech2text.DatasetManipulator import DatasetManipulator as DaMa


if len(sys.argv) < 2:
    print('No parameter given.')
    exit(1)

dataset_folder = sys.argv[1]  # ../datasets/cpm_dataset

am = DaMa(dataset_folder)
am.parallel_cut_all_pairs(n_jobs=1)
am.csv_from_cut_folder()
am.csv_generate_deepspeech(os.path.join(DaMa.OS_SEP.join(dataset_folder.rstrip(DaMa.OS_SEP).split(DaMa.OS_SEP)[:-1]),
                                        DaMa.CUT_DATASET_FOLDER_NAME))
