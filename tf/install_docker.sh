#!/bin/bash
cd
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt-get install ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo mkdir /var/lib/docker
sudo mkdir /mnt/docker
sudo mount --rbind /mnt/docker /var/lib/docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
