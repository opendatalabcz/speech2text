# speech2text

This project's purpose is to preprocess the Czech Parliament Meetings audio recordings ([here](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0005-CF9C-4?fbclid=IwAR3KCJk-TtYHq6VtcjZlDdL_phswtDMtU_VeaCgyRfC-dHjvrYrsd1amrzg)) and feed the proprocessed data to a neural network. I'm using Mozilla's [DeepSpeech](https://github.com/mozilla/DeepSpeech) project as my neural network model.

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

\*Choose an existing folder from the host machine to be mounted into the container. That's where you'll keep your datasets, exported models, etc. On the other hand, be careful what you put there since **you'll have to set high (777) right for the folder and all it's contents** from the host machine for the container to be able to manage files in the folder. You can find the shared directory at /opt/shared in the container afterwards.

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
