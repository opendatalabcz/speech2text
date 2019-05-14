from speech2text.DatasetManipulator import DatasetManipulator as DM


dataset_folder = "datasets/czech_parliament_meetings"
am = DM(dataset_folder)


# am.plot_wav_file('datasets/cpm_cut/0/1.wav')
am.parallel_cut_all_pairs()
