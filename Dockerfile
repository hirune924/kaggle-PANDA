FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

MAINTAINER "@hirune924"

RUN apt-get update --fix-missing && apt-get install -y git libopencv-dev\
    && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
    
RUN pip install pandas scikit-image albumentations opencv-python \
                scikit-learn pytorch-lightning neptune-client imagecodecs tifffile
                
