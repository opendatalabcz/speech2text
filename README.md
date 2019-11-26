# speech2text

This project's purpose is to preprocess the Czech Parliament Meetings (CPM) audio recordings ([here](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0005-CF9C-4?fbclid=IwAR3KCJk-TtYHq6VtcjZlDdL_phswtDMtU_VeaCgyRfC-dHjvrYrsd1amrzg)) and feed the proprocessed data to a neural network. I'm using Mozilla's [DeepSpeech](https://github.com/mozilla/DeepSpeech) project as my neural network model.

Project is designed to be used as a **docker** container, therefore both [Dockerfile](./Dockerfile) and [Makefile](./Makefile) are included.

## How to begin

Start by cloning this repo:
```
git clone https://github.com/opendatalabcz/speech2text.git
```

Now, it is necessary to also clone the DeepSpeech repo, since cloning it for each image build is rather impossible (aprox. 1.8 GB). That requires the git-lfs extension for git -- [here](https://git-lfs.github.com/). Clone it:
```
git clone https://github.com/mozilla/DeepSpeech.git
```

Now we have both repos next to each other. Now you should be ready to build a docker image. Let's switch to speech2text directory and use make to build the image (if you wish to change the image/container names, head to the Makefile and look for "I_NAME"/"C_NAME" parameters):
```
cd speech2text
make build
```

The image should contain everything necessary to train the model, export it and make inference with the exported model. After the image is built, we can again use make to run the container. You can (**and should**) check the [Makefile](./Makefile) to make your own changes based on the setup of your machine/cluster. It's very probable that the default settings are inappropriate for your setup. You should check:
* RAM limit ("RAM_LIMIT" parameter),
* GPU units to use ("GPU" parameter),
* Mounted host folder ("HOST_SHARED_DIR" parameter) - more info below\*

\*Choose an existing folder from the host machine to be mounted into the container. That's where you'll keep your datasets, exported models, etc. On the other hand, be careful what you put there since **you'll have to set high (777) rights for the folder and all it's contents** from the host machine for the container to be able to manage files in the folder. You can find the shared directory at /opt/shared in the container afterwards.

There are also two ports tunneled to the host machine from the container for Jupyter (port 8888) and Tensorboard (port 6006). If you're working on a remote server, you'll also need to forward these two ports via SSH/PuTTY (or whatever else you like to use).

The run command is set to remove the container as soon as you exit it (--rm option), so feel free to remove this option from the Makefile if you feel like keeping the container after each session.

Now you're ready to run the container:
```
make run
```

### Docker container
You're in. Everything important for us happens in the /opt directory. Also, there's a Python3 virtual environment ready for us to use:
```
cd /opt
. ./venvs/deepspeech-train-venv/bin/activate
```
The container is based on **nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04** image. There are all the requirements for DeepSpeech project and this means also the required version of GPU Tensorflow and CUDA dependencies.

In the /opt directory, you should see:
* the *DeepSpeech* repository,
* *inference.sh* script we will use later on,
* *kenlm* directory containing binaries to build language models,
* *shared* directory from the host machine,
* *speech2text* directory, which is this repo and
* *venvs*, where is the currently used Python virtual environmnet

## Data preprocessing
I recommend creating a "datasets" directory in the \opt\shared directory where you'll copy the CPM dataset - something along these lines:
```
mkdir -p /opt/shared/datasets/cpm_dataset /opt/shared/models
```
which also creates *cpm_dataset* for the files from [this](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0005-CF9C-4?fbclid=IwAR3KCJk-TtYHq6VtcjZlDdL_phswtDMtU_VeaCgyRfC-dHjvrYrsd1amrzg) source also mentioned above and *models* directory for future model exports.

After you copy the files to the *cpm_dataset* folder, you can use the *main.py* to preprocess the dataset for our neural net. What does it do? There are several phases:
1. Check the provided directory and look for audio/xml pairs.
2. Cut the audio files based on their xml synchronization tags and create transription txt files with the same name (creates a directory next to the one provided - named *cpm_cut*).
> This is being run **in parallel(!)**, therefore be careful when there's limited RAM on you machine. It runs with 8 threads defaultly which consumes at least 12 GB of memory on it's own. Feel free to change the "n_jobs" parameter of "parallel_cut_all_pairs" funtion in the *main.py* file before running it.
3. Generate a generic csv file from the audio_cut/transcription pairs in the folder.
4. Based on the generic csv create another csv which can be consumed by DeepSpeech, filter audio pairs with empty transcripts.

So execute the Python program as follows (and modify the dataset path if it differs):
```
python speech2text/main.py shared/datasets/cpm_dataset
```
if it runs correctly, it should first list all found audio/xml pairs, then list all the completed pair cuts and finally announce the train/dev/test file distribution for the DeepSpeech csv. Example:
```
Found:
SOUND_110212_003546, SOUND_110325_010705, SOUND_110427_005836, SOUND_110428_005954, SOUND_110429_005820, SOUND_110504_005351, SOUND_110505_005813, SOUND_110506_005821, SOUND_110507_005803, SOUND_110511_005438, SOUND_110609_005958, SOUND_110611_010015, SOUND_110617_010946, SOUND_110618_010259, SOUND_110619_001010, SOUND_110714_010025, SOUND_110715_005912, SOUND_110831_005909
Finished cutting file id 1.
Finished cutting file id 2.
Finished cutting file id 0.
Finished cutting file id 7.
...
Finished cutting file id 17.
Finished cutting file id 16.
Excluded samples with empty transcript.
51615 samples out of 60194
Distribution:
train: 0.6985178727114211
dev:   0.2009299622202848
test:  0.1005521650682941
```
## Language model creation
Before we start training the model, we need to generate a language model (LM). We need a vocabulary (more like a corpus) -- that should be a file, where on each line is one complete sentence from the dataset environment.

I've used [this](https://github.com/Dl2oWn/steno/blob/master/Priprava_dat.ipynb) jupyter notebook to do the data scraping for me. It scrapes the website of Czech Parliament, downloads all available compressed stenoprotocols from Czech Parliaments Meetings, processes them and creates vocabulary.txt file with the desired format (our corpus).

After obtaining the corpus file, copy it to the shared (mounted) folder from the host machine. Now we can start generating necessary files. We'll start with *arpa* file:
```
cd /opt
./kenlm/build/bin/lmplz --text shared/vocabulary.txt --arpa shared/words.arpa --o 5
```

where *--text* parameter expect path to the corpus file, *--arpa* the output filename and *--o* the order of the language model to estimate. The order should be a lower number for smaller corpuses and vice versa - more info [here](https://kheafield.com/code/kenlm/estimation/). Now to generation of LM binary:
```
./kenlm/build/bin/build_binary shared/words.arpa \
                               shared/lm.binary
```

and finally generate trie:
```
./DeepSpeech/native_client_prebuilt/generate_trie shared/alphabet_cz.txt \
                                                  shared/lm.binary \
                                                  shared/trie
```

## Training your model
With the dataset preprocessed and ready, let's try out training. There is already a script which, apart from executing the DeepSpeech training procedure, also logs DS outputs, saves training configuration, properly names exported model and converts it to mem-mappable format (much faster for inference). Let's swith to the DeepSpeech dir:
```
cd /opt/DeepSpeech
```
