import os
import glob
import re
import pydub as pdb
import matplotlib.pyplot as plt
import numpy as np
import wave
import multiprocessing as mp
import string
import random

try:
    import xml.etree.cElementTree as ElemT
except ImportError:
    import xml.etree.ElementTree as ElemT


class DatasetManipulator:

    FILE_ENCODING = 'utf-8-sig'
    OS_SEP = os.path.sep
    CUT_DATASET_FOLDER_NAME = "cpm_cut"
    CUT_CSV_NAME = "data.csv"
    SAMPLING_RATE = 16000

    audio_files = []
    annotation_files = []

    cut_dataset_path = ''

    def __init__(self, dataset_folder):
        self.cut_dataset_path = os.path.join(
            self.OS_SEP.join(dataset_folder.rstrip(self.OS_SEP).split(self.OS_SEP)[:-1]),
            self.CUT_DATASET_FOLDER_NAME)
        for r, d, f in os.walk(dataset_folder):
            for file in f:
                if '.wav' in file:
                    self.audio_files.append(os.path.join(r, file))
                if '.trs' in file:
                    self.annotation_files.append(os.path.join(r, file))

        if len(self.audio_files) == 0 or len(self.annotation_files) == 0:
            print("No viable files present in {}".format(dataset_folder))
            return

        self.audio_files.sort()
        self.annotation_files.sort()
        print("Found:")
        for i, f in enumerate(self.audio_files):
            if i == len(self.audio_files) - 1:
                print(self.audio_files[i].split(self.OS_SEP)[-1].split('.')[0])
            else:
                print(self.audio_files[i].split(self.OS_SEP)[-1].split('.')[0] + ', ', end='')

    def cut_audio_pair(self, pair_id):

        pair = self.get_pair_by_id(pair_id)
        if os.path.exists(self.cut_dataset_path):
            pattern = re.escape(str(pair_id)) + r'_.*'
            for f in os.listdir(self.cut_dataset_path):
                if re.search(pattern, f):
                    os.remove(os.path.join(self.cut_dataset_path, f))
        else:
            os.makedirs(self.cut_dataset_path)

        counter = 0
        last_time = 0
        file_id_path = os.path.join(self.cut_dataset_path, str(pair_id) + '_' + str(counter))
        first_file_entry = True
        pronounce_word = ''

        for elem in pair[1].iter():
            if elem.tag == 'Event' and elem.get('type') == 'pronounce' and elem.get('extent') == 'begin':
                pronounce_word = elem.get('desc')
            if elem.tag == 'Sync':
                first_file_entry = True
                time_str = elem.get('time')
                time_ms = int(float(time_str)*1000)
                audio_segment = pair[0][last_time:time_ms]
                audio_segment = audio_segment.set_frame_rate(self.SAMPLING_RATE)
                audio_segment.export(file_id_path + '.wav', format='wav')
                open(file_id_path + '.txt', 'a+', encoding=self.FILE_ENCODING).close()
                counter += 1
                file_id_path = os.path.join(self.cut_dataset_path, str(pair_id) + '_' + str(counter))
                last_time = time_ms
            for elem_text in [elem.text, elem.tail]:
                if elem_text and elem_text.strip():
                    file = open(file_id_path + '.txt', 'a+', encoding=self.FILE_ENCODING)
                    if pronounce_word:
                        if first_file_entry:
                            file.write(pronounce_word)
                        else:
                            file.write(' ' + pronounce_word)
                        pronounce_word = ''
                    else:
                        sent = ' '.join(elem_text.strip().replace(',', ' ').split())\
                                  .translate(str.maketrans('', '', string.punctuation))
                        if first_file_entry:
                            file.write(sent)
                        else:
                            file.write(' ' + sent)
                    file.close()
                    first_file_entry = False
        print('Finished cutting file id {}.'.format(pair_id))

    def csv_from_cut_folder(self):
        if not os.path.exists(self.cut_dataset_path):
            print('Cut folder doesn\'t exist.')
            return

        csv_file = open(os.path.join(self.cut_dataset_path, self.CUT_CSV_NAME), 'w+', encoding=self.FILE_ENCODING)
        csv_file.write('file,text\n')

        for file in os.listdir(self.cut_dataset_path):
            if '.wav' in file:
                file_name = file.split(self.OS_SEP)[-1]
                file_name_wo_ext, extension = os.path.splitext(file)
                with open(os.path.join(self.cut_dataset_path, file_name_wo_ext + '.txt'),
                          'r', encoding=self.FILE_ENCODING) as f:
                    label_str = f.read()
                csv_file.write('"{}","{}"\n'.format(file_name, label_str))

        csv_file.close()

    @staticmethod
    def csv_generate_deepspeech(folder, reduce_data=1.0):
        if not os.path.exists(folder):
            print('Folder doesn\'t exist.')
            return

        if 0 >= reduce_data > 1:
            print("reduce_data parameter should be a double "
                  "between 0 (exclusive) and 1 (inclusive - default). Returning.")
            return

        csv_train = open(os.path.join(folder, "train.csv"), 'w+', encoding=DatasetManipulator.FILE_ENCODING)
        csv_test = open(os.path.join(folder, "test.csv"), 'w+', encoding=DatasetManipulator.FILE_ENCODING)
        csv_dev = open(os.path.join(folder, "dev.csv"), 'w+', encoding=DatasetManipulator.FILE_ENCODING)

        for file in [csv_train, csv_test, csv_dev]:
            file.write('wav_filename,wav_filesize,transcript\n')

        wav_list = glob.glob(os.path.join(folder, "*.wav"))

        alphanum_pattern = re.compile(r'([^\s\w]|_+)', re.UNICODE)

        wav_indices = random.sample(range(len(wav_list)), int(np.ceil(len(wav_list)*reduce_data)))

        train_cnt = test_cnt = dev_cnt = 0

        for i in wav_indices:
            name, _ = os.path.splitext(wav_list[i])
            with open(name + '.txt', encoding=DatasetManipulator.FILE_ENCODING) as f:
                transcript = f.read().lower()
            transcript = alphanum_pattern.sub('', str(transcript))

            if transcript == "" or transcript == " ":
                continue

            filesize = os.path.getsize(wav_list[i])
            abs_path = os.path.abspath(wav_list[i])

            d = random.randint(0, 9)

            if d <= 6:
                csv_train.write('{},{},{}\n'.format(abs_path, filesize, transcript))
                train_cnt += 1
            elif d <= 8:
                csv_dev.write('{},{},{}\n'.format(abs_path, filesize, transcript))
                dev_cnt += 1
            else:
                csv_test.write('{},{},{}\n'.format(abs_path, filesize, transcript))
                test_cnt += 1

        for file in [csv_train, csv_test, csv_dev]:
            file.close()
        sample_sum = train_cnt + test_cnt + dev_cnt
        print("Excluded samples with empty transcript.")
        print("{} samples out of {}\nDistribution:\ntrain: {}\ndev:   {}\ntest:  {}"
              .format(sample_sum, len(wav_list), train_cnt/sample_sum, dev_cnt/sample_sum, test_cnt/sample_sum))

    @staticmethod
    def plot_wav_file(filename):
        spf = wave.open(filename, 'r')
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, 'Int16')
        fr = spf.getframerate()

        if spf.getnchannels() == 2:
            print('File has more then one channel, returning...')
            return

        y_time = np.linspace(0, len(signal) / fr, num=len(signal))
        plt.figure(1)
        plt.title('Signal wave of {}'.format(filename.split('/')[-1]))
        plt.plot(y_time, signal)
        plt.show()

    def parallel_cut_all_pairs(self, n_jobs=3):
        mp.Pool(n_jobs).map(self.cut_audio_pair, range(0, self.num_loaded()))

    def get_pair_by_id(self, pair_id):
        if pair_id < 0 or pair_id > len(self.audio_files) - 1:
            raise ValueError("Out of bounds - must be integer between 0 and " + str(self.num_loaded()))

        return pdb.AudioSegment.from_wav(self.audio_files[pair_id]), ElemT.parse(self.annotation_files[pair_id]).getroot()

    def num_loaded(self):
        return len(self.audio_files)
