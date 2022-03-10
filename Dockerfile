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

# Unused packages and channels, for reference
#RUN conda config --add channels conda-forge pytorch
#RUN conda install torchvision
#RUN conda install cudatoolkit

# Packages in the Python source
# Using the conda channel as default except where noted
RUN conda install pytorch -n hipims -y
RUN conda install numpy -n hipims -y
RUN conda install matplotlib -n hipims -y
RUN conda install seaborn -n hipims -y
RUN conda install shapely -n hipims -y
RUN conda install rasterio -n hipims -y
RUN conda install geopandas -n hipims -y
RUN conda install earthpy -n hipims -y --channel conda-forge

# Additional packages needed
RUN conda install pyqt -n hipims -y
RUN conda install tqdm -n hipims -y
RUN conda install kiwisolver -n hipims -y
RUN conda install pysal -n hipims -y
RUN conda install pyproj -n hipims -y
RUN conda install rasterstats -n hipims -y
RUN conda install geopy -n hipims -y --channel conda-forge
RUN conda install cartopy -n hipims -y
RUN conda install contextily -n hipims -y --channel conda-forge
RUN conda install folium -n hipims -y --channel conda-forge
RUN conda install geojson -n hipims -y --channel conda-forge
RUN conda install mapboxgl -n hipims -y --channel conda-forge
RUN conda install hydrofunctions -n hipims -y --channel conda-forge
RUN conda install geocoder -n hipims -y --channel conda-forge
RUN conda install tweepy -n hipims -y --channel conda-forge

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
