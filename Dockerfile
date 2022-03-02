# Base CUDA on Ubuntu
FROM nvidia/cuda:10.1-devel-ubuntu18.04

# install anaconda and required packages for python scripts
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt update --fix-missing && \
    apt install -y wget bzip2 ca-certificates \
        libglib2.0-0 libxext6 libsm6 libxrender1

RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate objdet" >> ~/.bashrc

# Use python 3.6 due to syntax incompatibility with python 3.7+
RUN conda create -n hipims python=3.7 -y
RUN conda activate hipims

RUN conda install pytorch torchvision cudatoolkit
RUN conda install numpy matplotlib pyqt seaborn tqdm kiwisolver
RUN conda install rasterio pysal pyproj rasterstats geopy cartopy contextily earthpy folium
RUN conda install geojson mapboxgl geocoder tweepy

# Set CUDA
ENV CUDA_ROOT /usr/local/cuda/bin

# get hipims code, input data, and python script to setup and run hipims model
RUN mkdir -p /hipims
WORKDIR /hipims

# create a data dir (this is where DAFNI will check for the data)
RUN mkdir /data
RUN mkdir /data/outputs

# copy files over
COPY cuda /hipims/cuda 
COPY pythonHipims /hipims/pythonHipims
COPY Newcastle /hipims/Newcastle 

#RUN pwd
# compile hipims model
WORKDIR /hipims/hipims_apps/cuda
RUN python setup.py install

#Mount output directories. Must be container directory
VOLUME /hipims/Outputs

# Entrypoint, comment out either one of the CMD instructions
WORKDIR /hipims/Newcastle
CMD python3 singleGPU_example.py
