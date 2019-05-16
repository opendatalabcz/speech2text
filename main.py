from speech2text.DatasetManipulator import DatasetManipulator as DM
from speech2text.Experiment import experiment

# alphabet='aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž'


# dataset_folder = "datasets/czech_parliament_meetings"
# am = DM(dataset_folder)

# am.plot_wav_file('datasets/cpm_cut/0_1.wav')
# am.parallel_cut_all_pairs(n_jobs=3)
# am.csv_from_cut_folder()
#
# dataset = pd.read_csv('datasets/cpm_cut/data.csv')
# for _, row in dataset.sample(frac=1).iterrows():
#     print('|', _, '|', '/', row, '/')

experiment()
