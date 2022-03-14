###############################################################################
# Base image - CUDA on Ubuntu
###############################################################################
FROM nvidia/cuda:10.2-devel-ubuntu18.04

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

# Get and install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN conda update -n base -c defaults conda
# Use conda-forge and strict channel priority (setup .condarc for this)
RUN echo -e \
"channel_priority: strict\n\
channels:\n\
  - conda-forge\n\
  - defaults" > ~/.condarc

# Requires Python 3.7 - all requirements are in hipims-environment.yml
# Does NOT work with Python 3.8+, specifically with rasterio module
COPY hipims-environment.yml .
RUN conda env create -f hipims-environment.yml --debug

# Need this if we want to use RUN commands in the proper environment
#SHELL ["conda", "run", "-n", "hipims", "/bin/bash", "-c"]

# Unused packages and channels, for reference
#RUN conda config --add channels conda-forge pytorch
#RUN conda install torchvision
#RUN conda install cudatoolkit

# Packages in the Python source
# Using the conda channel as default except where noted
#RUN CONDA_CUDA_OVERRIDE="10.2" conda install "pytorch==1.10.2=cuda102*" -n hipims --channel conda-forge -y
#UN conda install numpy=1.21.5
#UN conda install matplotlib=3.5.1 -n hipims -y
#UN conda install seaborn=0.11.2 -n hipims -y
#UN conda install shapely=1.8.0 -n hipims -y
#UN conda install rasterio=1.2.10 -n hipims -y
#UN conda install geopandas=0.10.2 -n hipims -y
#RUN conda install earthpy=0.9.4 -n hipims -y --channel conda-forge
#UN conda install earthpy=0.9.4 -n hipims -y

# Additional packages needed
#UN conda install pyqt=5.12.3 -n hipims -y
#UN conda install tqdm=4.62.3 -n hipims -y
#UN conda install kiwisolver=1.3.2 -n hipims -y
#UN conda install pysal=2.6.0 -n hipims -y
#UN conda install pyproj=3.2.1 -n hipims -y
#UN conda install rasterstats=0.16.0 -n hipims -y
#RUN conda install geopy=2.2.0 -n hipims -y --channel conda-forge
#UN conda install geopy=2.2.0 -n hipims -y
#UN conda install cartopy=0.20.2 -n hipims -y
#RUN conda install contextily=1.2.0 -n hipims -y --channel conda-forge
#UN conda install contextily=1.2.0 -n hipims -y
#RUN conda install folium=0.12.1.post1 -n hipims -y --channel conda-forge
#UN conda install folium=0.12.1.post1 -n hipims -y
#RUN conda install geojson=2.5.0 -n hipims -y --channel conda-forge
#UN conda install geojson=2.5.0 -n hipims -y
#RUN conda install mapboxgl=0.10.2 -n hipims -y --channel conda-forge
#UN conda install mapboxgl=0.10.2 -n hipims -y
#RUN conda install hydrofunctions=0.2.1 -n hipims -y --channel conda-forge
#UN conda install hydrofunctions=0.2.1 -n hipims -y
#RUN conda install geocoder=1.38.1 -n hipims -y --channel conda-forge
#UN conda install geocoder=1.38.1 -n hipims -y
#RUN conda install tweepy=4.6.0 -n hipims -y --channel conda-forge
#UN conda install tweepy=4.6.0 -n hipims -y

# Set CUDA environment
ENV CUDA_ROOT /usr/local/cuda/bin

# Compile hipims model
WORKDIR /cuda
#RUN python setup.py install

# you should get back to the directory of 'singleGPU_example.py' at first 
# sorry I don't know the command
#RUN python singleGPU_example.py

# Entrypoint, comment out either one of the CMD instructions
WORKDIR /hipims/Newcastle
#CMD ["python", "singleGPU_example.py"]
CMD ["conda", "run", "--no-capture-output", "-n", "hipims", "/bin/bash"]
