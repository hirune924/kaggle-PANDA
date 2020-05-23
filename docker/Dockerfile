FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

ARG DEBIAN_FRONTEND=noninteractive

MAINTAINER "@hirune924"

RUN apt-get update --fix-missing && apt-get install -y git libopencv-dev\
    && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd .. && rm -rf apex

RUN pip install pandas scikit-image albumentations opencv-python \
                scikit-learn pytorch-lightning neptune-client imagecodecs tifffile \
                pretrainedmodels catalyst[all] segmentation-models-pytorch
