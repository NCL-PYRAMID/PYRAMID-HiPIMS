###############################################################################
# Base image - CUDA on Ubuntu
###############################################################################
FROM nvidia/cuda:10.1-devel-ubuntu18.04

###############################################################################
# Anaconda setup
###############################################################################

# Relevant environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Update apt and install appropriate packages
RUN apt update --fix-missing
RUN apt install -y wget
RUN apt upgrade -y

# Get and install Anaconda, add conda environment setup to end of ./bashrc
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN conda update -n base -c defaults conda

# Use python 3.6 due to syntax incompatibility with python 3.7+
# But this uses Python 3.7?
RUN conda create -n hipims python=3.7 -y
RUN conda init bash
RUN echo "conda activate hipims" >> ~/.bashrc

# 1. Do we need conda-forge?
#RUN conda config --add channels conda-forge 
# 2. Do we need to use the "pytorch" channel?
#RUN conda install pytorch -c pytorch
#RUN conda install torchvision -c pytorch
#RUN conda install cudatoolkit -c pytorch
#RUN conda install torchvision
#RUN conda install cudatoolkit

# 3. Definitely in the Python source
##RUN conda install pytorch
#RUN conda install numpy
#RUN conda install matplotlib
#RUN conda install seaborn
##RUN conda install shapely
RUN bash -c 'conda install rasterio'
##RUN conda install geopandas
##RUN conda install earthpy

# 4. What are all of these?
#RUN conda install pyqt
#RUN conda install tqdm
#RUN conda install kiwisolver
#RUN conda install pysal
#RUN conda install pyproj
#RUN conda install rasterstats
#RUN conda install geopy
#RUN conda install cartopy
#RUN conda install contextily
#RUN conda install folium
#RUN conda install geojson
#RUN conda install mapboxgl
#RUN conda install hydrofunctions 

# 5. What are these?
#RUN conda install geocoder tweepy


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
#COPY Newcastle /hipims/Newcastle 

#RUN pwd
# compile hipims model
WORKDIR /hipims/hipims_apps/cuda
#RUN python setup.py install

#Mount output directories. Must be container directory
VOLUME /hipims/Outputs

# Entrypoint, comment out either one of the CMD instructions
WORKDIR /hipims/Newcastle
#CMD python3 singleGPU_example.py
CMD bash
