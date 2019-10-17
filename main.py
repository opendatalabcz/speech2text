import sys

from speech2text.DatasetManipulator import DatasetManipulator as DM


if len(sys.argv) < 2:
    print('No parameter given.')
    exit(1)

dataset_folder = sys.argv[1]  # ../datasets/cpm_dataset

am = DM(dataset_folder)

# am.plot_wav_file('datasets/cpm_cut/0_1.wav')
am.cut_audio_pair(0)
# am.csv_from_cut_folder()
#
# dataset = pd.read_csv('datasets/cpm_cut/data.csv')
# for _, row in dataset.sample(frac=1).iterrows():
#     print('|', _, '|', '/', row, '/')
# alphabet='aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž'
