import sys, getopt
import os

from speech2text.DatasetManipulator import DatasetManipulator as DaMa

dataset_dir = ''
n_jobs = 1
try:
    opts, args = getopt.getopt(sys.argv, "hi:n", ["inputdir=","njobs="])
except getopt.GetoptError:
    print 'main.py -i <input_dir> -n <n_parallel_jobs>'
    sys.exit(2)
for opt, arg in opts:
    if opt == 'h':
        print 'main.py -i <input_dir> -n <n_parallel_jobs>'
        sys.exit()
    elif opt in ("-i", "--inputdir"):
        dataset_dir = arg
    elif opt in ("-n", "--njobs"):
        n_jobs = arg
        

am = DaMa(dataset_dir)
am.parallel_cut_all_pairs(n_jobs=n_jobs)
am.csv_from_cut_folder()
am.csv_generate_deepspeech(os.path.join(DaMa.OS_SEP.join(dataset_dir.rstrip(DaMa.OS_SEP).split(DaMa.OS_SEP)[:-1]),
                                        DaMa.CUT_DATASET_FOLDER_NAME))
