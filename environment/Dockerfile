# hash:sha256:04a0bfb0988643cd27b0a87cb32142defb2425312d82099a3ad9b0c77c7c4e93
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -U --no-cache-dir \
    scipy==1.12.0 \
    opencv-python==4.9.0.80 \
    gdown==5.1.0 \
    matplotlib==3.8.0 \
    h5py==3.10.0 \
    tensorflow==2.9