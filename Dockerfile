# NVIDIA CUDA RUNTIME
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

#DeepSpeech tags
ARG DS_TAG_VER="0.6.0"
ARG DS_TAG_ALP="12"

ARG DS_CHECKOUT="v${DS_TAG_VER}-alpha.${DS_TAG_ALP}"
ARG DS_PYTHON="${DS_TAG_VER}a${DS_TAG_ALP}"

# UBUNTU DEPENDENCIES
RUN apt-get update && apt-get install -y \
			wget \
			git \
			git-lfs \
			nano \
            vim \
            curl \
            python3 \
            python3-pip \
			sox \
			libsox-fmt-mp3 \
			locales \
			dos2unix \
			virtualenv \
            build-essential \
			cmake \
			libboost-all-dev \
			liblzma-dev \
			libbz2-dev \
            software-properties-common
RUN git lfs install
RUN apt-get clean
 
# Custom CUDA Library paths
# ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64

# Copy the DeepSpeech repo and update it
COPY ./DeepSpeech /opt/DeepSpeech
RUN cd /opt/DeepSpeech && git pull && git checkout $DS_CHECKOUT
COPY ./speech2text/train_custom.sh /opt/DeepSpeech/train_custom.sh
COPY ./speech2text/inference.sh /opt/inference.sh

# Download and install KenLM toolkit
RUN cd opt/ && \
    wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz && \
    mkdir kenlm/build && \
    cd kenlm/build && \
    cmake .. && \
    make -j2

# Clone dataset preprocessing tools
RUN cd /opt && git clone https://github.com/opendatalabcz/speech2text.git

# Virtual environmnet
ENV VIRTUAL_ENV=/opt/venvs/deepspeech-train-venv
RUN virtualenv -p python3 "$VIRTUAL_ENV"
ENV PATH_DOCKER_BACKUP="$PATH"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# DeepSpeech Python dependencies, pydub package and Jupyter
RUN pip3 install deepspeech-gpu==$DS_PYTHON
RUN pip3 install -r /opt/DeepSpeech/requirements.txt && \
    pip3 install pydub && \
    pip3 uninstall -y tensorflow && pip3 install 'tensorflow-gpu==1.14.0' && \
    pip3 install jupyterlab
RUN pip3 install $(python3 /opt/DeepSpeech/util/taskcluster.py --decoder)

# Env variable for DeepSpeech training failure
# ENV TF_FORCE_GPU_ALLOW_GROWTH true

# Revert PATH environmnet variable (exit venv)
ENV PATH="$PATH_DOCKER_BACKUP"

# Set Python IO encoding to utf-8
ENV PYTHONIOENCODING="utf-8"

# Set US-english UTF-8 as locale
# RUN locale-gen en_US.UTF-8 && export LANG=en_US.UTF-8

# Set access right for copied files
# CMD chmod u+x /opt/inference.sh /opt/DeepSpeech/train_custom.sh
