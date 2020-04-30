# NVIDIA CUDA RUNTIME
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

#DeepSpeech tags
ARG DS_TAG_VER="0.6.0"

ARG DS_CHECKOUT="v${DS_TAG_VER}"
ARG DS_PYTHON="${DS_TAG_VER}"

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
	python3-dev \
	sox \
	libsox-fmt-mp3 \
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

# Copy the DeepSpeech repo and update it
COPY ./DeepSpeech /opt/DeepSpeech
RUN cd /opt/DeepSpeech && \
    git pull && \
    git checkout $DS_CHECKOUT

# Download and install KenLM toolkit
RUN cd opt/ && \
    wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz && \
    mkdir kenlm/build && \
    cd kenlm/build && \
    cmake .. && \
    make -j2

# Clone dataset preprocessing tools and scripts, give proper rights
RUN cd /opt && \ 
    git clone https://github.com/opendatalabcz/speech2text.git
RUN cp /opt/speech2text/inference.sh /opt/ && \
    chmod u+x /opt/inference.sh
RUN cp /opt/speech2text/train_custom.sh /opt/DeepSpeech/ && \
    chmod u+x /opt/DeepSpeech/train_custom.sh

# Virtual environmnet
ENV VIRTUAL_ENV=/opt/venvs/deepspeech-train-venv
RUN virtualenv -p python3 "$VIRTUAL_ENV"
ENV PATH_DOCKER_BACKUP="$PATH"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# DeepSpeech Python dependencies, pydub package and Jupyter
RUN pip3 install deepspeech-gpu==$DS_PYTHON
RUN pip3 install -r /opt/DeepSpeech/requirements.txt && \
    pip3 install pydub && \
    pip3 uninstall -y tensorflow && \
    pip3 install 'tensorflow-gpu==1.14.0' && \
    pip3 install jupyterlab
RUN pip3 install $(python3 /opt/DeepSpeech/util/taskcluster.py --decoder)

# Download the pb->pbmm converter and trie generator
RUN cd /opt/DeepSpeech && \
    python util/taskcluster.py --source "tensorflow" \
			       --branch "r1.14" \
			       --artifact "convert_graphdef_memmapped_format" \
			       --target native_client_bin && \
    chmod u+x /opt/DeepSpeech/native_client_bin/convert_graphdef_memmapped_format

RUN cd /opt/DeepSpeech && \
    python util/taskcluster.py --target ./native_client_prebuilt

# Env variable for DeepSpeech training failure
# ENV TF_FORCE_GPU_ALLOW_GROWTH true

# Revert PATH environmnet variable (exit venv)
ENV PATH="$PATH_DOCKER_BACKUP"

# Set Python IO encoding to utf-8
ENV PYTHONIOENCODING="utf-8"
