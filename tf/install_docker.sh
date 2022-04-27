#!/bin/bash
# Shell script to set up Docker on an Azure VM as a "azurerm_virtual_machine_extension"
# NOTE: Check limits to embedded Terraform scripts using base64 encoding

# sanity check
echo 'hello' > /home/pyramidtestuser/test.txt

# apt
cd
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt-get install ca-certificates curl gnupg lsb-release

# docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo mkdir /var/lib/docker
sudo mkdir /mnt/docker
sudo mount --rbind /mnt/docker /var/lib/docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# NCT for docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker
