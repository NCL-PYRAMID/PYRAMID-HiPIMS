###############################################################################
# Base image - CUDA on Ubuntu
###############################################################################
FROM nvidia/cuda:10.1-devel-ubuntu18.04

###############################################################################
# Anaconda setup
# See: https://pythonspeed.com/articles/activate-conda-dockerfile/
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

# Requires Python 3.7
# Does NOT work with Python 3.8+, specifically with rasterio module
COPY environment.yml .
RUN conda env create -f environment.yml
#SHELL ["conda", "run", "-n", "hipims", "/bin/bash", "-c"]

# 1. Do we need conda-forge?
#RUN conda config --add channels conda-forge 
# 2. Do we need to use the "pytorch" channel?
#RUN conda install pytorch -c pytorch
#RUN conda install torchvision -c pytorch
#RUN conda install cudatoolkit -c pytorch
#RUN conda install torchvision
#RUN conda install cudatoolkit

# 3. Definitely in the Python source
RUN conda install pytorch -n hipims -y
RUN conda install numpy -n hipims -y
RUN conda install matplotlib -n hipims -y
RUN conda install seaborn -n hipims -y
RUN conda install shapely -n hipims -y
RUN conda install rasterio -n hipims -y
RUN conda install geopandas -n hipims -y
# earthpy is only available from conda-forge
RUN conda install earthpy -n hipims -y -c conda-forge

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
CMD ["python", "singleGPU_example.py"]
#CMD ["conda", "run", "--no-capture-output", "-n", "hipims", "/bin/bash"]
