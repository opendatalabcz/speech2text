import os, re
import shutil
import pydub as pdb
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import multiprocessing as mp
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class DatasetManipulator:

    CUT_DATASET_PATH = "datasets/cpm_cut"

    audio_files = []
    annotation_files = []

    def __init__(self, dataset_folder):
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
                print(self.audio_files[i].split('/')[-1].split('.')[0])
            else:
                print(self.audio_files[i].split('/')[-1].split('.')[0] + ', ', end='')

    def cut_audio_pair(self, pair_id):
        pair = self.get_pair_by_id(pair_id)

        if os.path.exists(self.CUT_DATASET_PATH):
            pattern = re.escape(str(pair_id)) + r'_.*'
            for f in os.listdir(self.CUT_DATASET_PATH):
                if re.search(pattern, f):
                    os.remove(os.path.join(self.CUT_DATASET_PATH, f))
        else:
            os.makedirs(self.CUT_DATASET_PATH)

        counter = 0
        last_time = 0
        file_id_path = os.path.join(self.CUT_DATASET_PATH, str(pair_id) + '_' + str(counter))
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
                audio_segment.export(file_id_path + '.wav', format='wav')
                open(file_id_path + '.txt', 'a+').close()
                counter += 1
                file_id_path = os.path.join(self.CUT_DATASET_PATH, str(pair_id) + '_' + str(counter))
                last_time = time_ms
            for elem_text in [elem.text, elem.tail]:
                if elem_text and elem_text.strip():
                    file = open(file_id_path + '.txt', 'a+')
                    if pronounce_word:
                        if first_file_entry:
                            file.write(pronounce_word)
                        else:
                            file.write(' ' + pronounce_word)
                        pronounce_word = ''
                    else:
                        if first_file_entry:
                            file.write(' '.join(elem_text.strip().replace(',', ' ').split()))
                        else:
                            file.write(' ' + ' '.join(elem_text.strip().replace(',', ' ').split()))
                    file.close()
                    first_file_entry = False

    @staticmethod
    def plot_wav_file(filename):
        spf = wave.open(filename, 'r')
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, 'Int16')
        fr = spf.getframerate()

        if spf.getnchannels() == 2:
            print('Just mono files')
            sys.exit(0)

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
        return pdb.AudioSegment.from_wav(self.audio_files[pair_id]), ET.parse(self.annotation_files[pair_id]).getroot()

    def num_loaded(self):
        return len(self.audio_files)
