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

# Need this if we want to use RUN commands in the proper environment
#SHELL ["conda", "run", "-n", "hipims", "/bin/bash", "-c"]

# Unused packages and channels, for reference
#RUN conda config --add channels conda-forge pytorch
#RUN conda install torchvision
#RUN conda install cudatoolkit

# Packages in the Python source
# Using the conda channel as default except where noted
RUN conda install pytorch=1.8.1 -n hipims -y
RUN conda install numpy=1.19.2 -n hipims -y
RUN conda install matplotlib=3.5.1 -n hipims -y
RUN conda install seaborn=0.11.2 -n hipims -y
RUN conda install shapely=1.7.1 -n hipims -y
RUN conda install rasterio=1.1.0 -n hipims -y
RUN conda install geopandas=0.9.0 -n hipims -y
RUN conda install earthpy=0.9.4 -n hipims -y --channel conda-forge

# Additional packages needed
RUN conda install pyqt=5.9.2 -n hipims -y
RUN conda install tqdm=4.62.3 -n hipims -y
RUN conda install kiwisolver=1.3.2 -n hipims -y
RUN conda install pysal=1.14.4.post1 -n hipims -y
RUN conda install pyproj=2.6.1.post1 -n hipims -y
RUN conda install rasterstats=0.14.0 -n hipims -y
RUN conda install geopy=2.2.0 -n hipims -y --channel conda-forge
RUN conda install cartopy=0.18.0 -n hipims -y
RUN conda install contextily=1.2.0 -n hipims -y --channel conda-forge
RUN conda install folium=0.12.1.post1 -n hipims -y --channel conda-forge
RUN conda install geojson=2.5.0 -n hipims -y --channel conda-forge
RUN conda install mapboxgl=0.10.2 -n hipims -y --channel conda-forge
RUN conda install hydrofunctions=0.2.1 -n hipims -y --channel conda-forge
RUN conda install geocoder=1.38.1 -n hipims -y --channel conda-forge
RUN conda install tweepy=4.6.0 -n hipims -y --channel conda-forge

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
