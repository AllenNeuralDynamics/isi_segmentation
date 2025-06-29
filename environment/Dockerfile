# hash:sha256:f378458a7d6033e9c9ea75c76e80c6caa1c92da4b98fcfb55e1a8133cc653d15
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg=7:4.2.7-0ubuntu0.1 \
        libsm6=2:1.2.3-1 \
        libxext6=2:1.3.4-0ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.4.0 \
    gdown==5.2.0 \
    h5py==3.14.0 \
    imageio==2.37.0 \
    matplotlib==3.9.4 \
    opencv-python==4.11.0.86 \
    scipy==1.13.1 \
    tensorflow==2.19.0 \
    uv==0.7.15

COPY postInstall /
RUN /postInstall
