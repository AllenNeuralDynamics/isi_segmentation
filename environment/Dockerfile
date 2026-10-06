# hash:sha256:d2cf030d1ab1136dd3eb3f86005b52869e885a0e21d96b6c9222ae27767145ee
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg=7:4.2.7-0ubuntu0.1 \
        libsm6=2:1.2.3-1 \
        libxext6=2:1.3.4-0ubuntu1 \
        vim=2:8.1.2269-1ubuntu5.32 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.4.0 \
    aind-metadata-upgrader==0.0.25 \
    gdown==5.2.0 \
    h5py==3.14.0 \
    imageio==2.37.0 \
    matplotlib==3.9.4 \
    opencv-python==4.11.0.86 \
    scipy==1.13.1 \
    tensorflow==2.19.0 \
    uv==0.7.15

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
