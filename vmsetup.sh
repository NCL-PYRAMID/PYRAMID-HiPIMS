#!/bin/bash
###############################################################################
# Configuration script for setting up Azure VM components
# 
# Robin Wardle, May 2022
###############################################################################

# Assume that the VM uses Azure Extension for installing NVidia CUDA Toolkit
# Wait until the toolkithas finished installing
if [ 1 ]
then
while [ $(ps aux | grep -i apt | wc -l) -gt 1 ]
do sleep 10
echo 'Still installing CUDA toolkit ...'
done
echo 'CUDA toolkit installed'
fi

# Update packages before continuing
cd
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker's GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /usr/share/keyrings/docker-archive-keyring.gpg
# Set up the stable repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Create Docker storage under /mnt as / is too small a partition
sudo mkdir /var/lib/docker
sudo mkdir /mnt/docker
sudo mount --rbind /mnt/docker /var/lib/docker

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
sudo docker run hello-world

# Test
nvidia-smi

# Install the NVidia Container Toolkit for Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# Test
sudo docker run -it --gpus all nvidia/cuda:11.4.0-base-ubuntu20.04 nvidia-smi
