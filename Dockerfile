###############################################################################
# Base image - CUDA on Ubuntu
###############################################################################
FROM nvidia/cuda:11.0.3-devel-ubuntu18.04

###############################################################################
# Anaconda setup
# See: https://pythonspeed.com/articles/activate-conda-dockerfile/
###############################################################################

# Relevant environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# App working directory
WORKDIR /hipims

# Update apt and install appropriate packages
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update --fix-missing
RUN apt install -y wget
RUN apt upgrade -y

# Get and install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN conda update -y -n base -c defaults conda
# Use conda-forge and strict channel priority (setup .condarc for this)
RUN echo -e \
"channel_priority: strict\n\
channels:\n\
  - conda-forge\n\
  - defaults" > ~/.condarc

# Create himpims environment and set CUDA root
# Requires Python 3.7 - all requirements are in hipims-environment.yml
# Does NOT work with Python 3.8+, specifically with rasterio module
COPY hipims-environment.yml .
RUN conda env create -f hipims-environment.yml
#ENV CUDA_ROOT /usr/local/cuda/bin

# Need this if we want to use RUN commands in the proper environment
SHELL ["conda", "run", "-n", "hipims", "--no-capture-output", "/bin/bash", "-c"]

# Unused packages and channels, for reference
#RUN conda config --add channels conda-forge pytorch
#RUN conda install torchvision
#RUN conda install cudatoolkit

# Copy files to container
# DO NOT copy the whole directory - this runs the risk of copying existing
# saved Docker images and test data files. Also, the Terraform and Docker
# build files are unneccessary
WORKDIR /hipims
COPY ./ .

# Compile hipims model
WORKDIR /hipims/cuda
RUN python setup.py install

# Entrypoint, comment out either one of the CMD instructions
ENV HIPIMS_PLATFORM="docker"
WORKDIR /hipims
CMD ["conda", "run", "-n", "hipims", "--no-capture-output", "python", "singleGPU_example.py"]
#CMD ["conda", "run", "--no-capture-output", "-n", "hipims", "/bin/bash"]
