# PYRAMID-HiPIMS
CUDA-enabled HiPIMS for DAFNI

## About

HiPIMS standards for High-Performance Integrated hydrodynamic Modelling System. It uses state-of-art numerical schemes (Godunov-type finite volume) to solve the 2D shallow water equations for flood simulations. To support high-resolution flood simulations, HiPIMS is implemented on multiple GPUs (Graphics Processing Unit) using CUDA/C++ languages to achieve high-performance computing. 

### Project Team
Xue Tong, Loughborough University ([x.tong2@lboro.ac.uk](mailto:x.tong2@lboro.ac.uk))  
Elizabeth Lewis, Newcastle University  ([elizabeth.lewis2@newcastle.ac.uk](mailto:elizabeth.lewis2@newcastle.ac.uk))  

### RSE Contact
Robin Wardle  
RSE Team, Newcastle Data  
Newcastle University NE1 7RU  
([robin.wardle@newcastle.ac.uk](mailto:robin.wardle@newcastle.ac.uk))  


## Getting Started

### Prerequisites
HiPIMs is a CUDA-enabled application and requires NVidia GPU hardware 
and drivers to run, although it can be compiled with the NVidia CUDA Toolkit without the presence of NVidia hardware. The CUDA Toolkit contains the necessary libraries, header files and configuration information for the compilation of CUDA-enabled applications. The HiPMS deployment requires the following software to be installed on the build machine. These instructions assume a Debian-based Linux distribution and were developed under Ubuntu 20.04.

- a Linux operating system for build and installation with `root` access through `sudo`
- `apt` package management tool (or equivalent)
- `wget`
- GCC and G++ compilers
- The CUDA 10.2 Toolkit
- Anaconda Python environment management tool
- Python 3.7

The component requirements are described in detail in the following sections.

#### `apt` package management tool and `wget`
`apt` will be part of the Linux distribution. `wget` should also be present, but if not it can be installed using
```
sudo apt update
sudo apt install wget
```

#### GCC and G++ versions
CUDA 10.2 Toolkit requires GCC and G++ for its installation. GCC versions later than 8 are not supported by CUDA 10.2. This deployment uses GCC 7. Under Linux it is possible to select compiler versions using `update-alternatives`. To install GCC 7 as the default compiler under Linux, install and select it with the following commands:
```
sudo apt update
sudo apt install -y gcc-7 g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 10
```
The `10` at the end of the `update-alternatives` command is the compiler priority, with a higher number being a higher priority. Any existing compiler is likely to be of priority equal to its version number; after installation, check the installed compilers and their priorites with
```
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```
The priorites and default for the CUDA 10.2 installation should be highest for GCC 7.

#### Installing the CUDA Toolkit and Drivers
HiPIMS should work with the latest CUDA Toolkit installation but this deployment has been developed with CUDA 10.2. CUDA is open-source software available from NVidia and the installer includes:
- hardware drivers
- CUDA Toolkit
- CUDA examples
- CUDA demo suite
- CUDA documentation

There are three sources of of the CUDA installers of interest:
- [The latest version](https://developer.nvidia.com/cuda-downloads)
- [All archives](https://developer.nvidia.com/cuda-toolkit-archive)
- [CUDA 10.2 archive](https://developer.nvidia.com/cuda-10.2-download-archive)

If your build machine does not already have CUDA installed or if your CUDA version is older than version 10.2, then navigate to the [CUDA 10.2 archive](https://developer.nvidia.com/cuda-10.2-download-archive) and download the appropriate **runfile** for your system. Navigate to an appropriate directory and download the installer file and two patches using `wget` in a terminal window.
```
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/1/cuda_10.2.1_linux.run
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/2/cuda_10.2.2_linux.run
```
The CUDA Toolkit needs to be installed, and, optionally, if your hardware is NVidia GPU-equipped, the drivers themselves. The installer is run using either
```
sudo sh cuda_10.2.89_440.33.01_linux.run
```
for the interactive menu version where the installation options can be selected, or, alternatively in a non-interactive mode with
```
sudo sh cuda_10.2.89_440.33.01_linux.run --silent --toolkit --driver
```
omitting `--driver` if only the Toolkit is needed, i.e. if you do not have NVidia GPU hardware and wish to compile the application only. Other installation options are described in the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Having installed the main CUDA Toolkit, next install the patches:
```
sudo sh cuda_10.2.1_linux.run --silent --toolkit
sudo sh cuda_10.2.1_linux.run --silent --toolkit
```
These can also be installed interactively without the `--silent` and `--toolkit` flags if desired.

Finally you will need to add the following to your shell environment variables:
- `PATH` should include `/usr/local/cuda-10.2/bin`.
- `LD_LIBRARY_PATH` should include `/usr/local/cuda-10.2/lib64`. Alternatively, add `/usr/local/cuda-10.2/lib64` to `/etc/ld.so.conf` and run `ldconfig` as root

This can be done for the current shell as
```
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
```
These lines can also be added to your `~/.bashrc` file (or other `rc` file if using a different Linux shell) and issuing a "`source ~/.bashrc`" command.

#### Uninstalling the CUDA Toolkit
If necessary, uninstall the CUDA Toolkit by running
```
cd /usr/local/cuda-10.2/bin
sudo cuda-uninstaller
```
Alternatively, the Toolkit can be uninstalled manually if necessary:
```
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" 
sudo apt-get --purge remove "*nvidia*"
sudo rm -rf /usr/local/cuda*
```

#### Anaconda installation
HiPIMS is built around PyTorch 1.2 or higher and uses a number of other Python support packages. [Anaconda](https://www.anaconda.com/) is used to manage the packages needed by HiPIMS, and the dependent packages are listed in an exported Anaconda environment file `hipims-environment.yml`.

_a. Install Anaconda_  
If not already present, Anaconda should be installed and updated. The main [Anaconda documentation](https://docs.anaconda.com/anaconda/install/index.html) describes the installation process but the process is summarised here. Download the [Anaconda installation file](https://www.anaconda.com/products/individual?modal=nucleus#linux) to your Downloads folder and install it:
```
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
cd ~/Downloads
sudo sh ~/Downloads/Anaconda3-2021.11-Linux-x86_64.sh
conda update -n base -c defaults conda
```
The name of the Anaconda `.sh` file may be diferent. Follow the prompts, consulting the documentation if necessary.

_b. Initialise Anaconda to use your preferred login shell_  
For `bash`, this is
```
conda init bash
```
You will need to log out of this shell and re-log in to ensure that Anaconda works correctly with the shell.

#### Anaconda environment configuration
The Anaconda environment is best created using the `hipims-environment.yml` file, which contains a description of all of the packages in the correct versions.
```
conda env create -f hipims-environment.yml -y
conda activate hipims
```

Alternatively, the following series of commands can create the environment using individual `conda create` instructions.

_a. Create and activate an Anaconda virtual environment with Python 3.7_  
Some of the packages used in the HiPIMS application are only available from `conda-forge`, so all packages will use `conda-forge` as the default installation channel.

```
conda create -c conda-forge -y --name hipims python=3.7
conda activate hipims
```
_b. Install pytorch, torchvision and the CUDA toolkit._  
```
conda install -c conda-forge -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
```
_c. Install the python packages used by HiPIMS._  
```
conda install -c conda-forge -y numpy
conda install -c conda-forge -y matplotlib
conda install -c conda-forge -y pyqt
conda install -c conda-forge -y seaborn
conda install -c conda-forge -y tqdm
conda install -c conda-forge -y kiwisolver
conda install -c conda-forge -y rasterio
conda install -c conda-forge -y pysal
conda install -c conda-forge -y pyproj
conda install -c conda-forge -y rasterstats
conda install -c conda-forge -y geopy
conda install -c conda-forge -y cartopy
conda install -c conda-forge -y contextily
conda install -c conda-forge -y earthpy
conda install -c conda-forge -y folium
conda install -c conda-forge -y geojson
conda install -c conda-forge -y mapboxgl
conda install -c conda-forge -y hydrofunctions
conda install -c conda-forge -y geocoder
conda install -c conda-forge -y tweepy
```

### Installation
#### HiPIMS CUDA Library Compilation and Installation
The main HiPIMS library consists of `.cpp` and `.cu` code that must be compiled. `setuptools` is used to drive the compilation and installation of the library. The code and installation script is in the `cuda` subdirectory.

During the compilation process the environment variable `CUDA_HOME` is used. Assuming that CUDA 10.2 was installed in the default location then a symbolic link `/usr/local/cuda` is created which points to `/usr/local/cuda-10.2`. The compilation process will default to using `CUDA_HOME=/usr/local/cuda` and will proceed without problem; however, if you installed CUDA to a different directory and / or the symbolic link from `/usr/local/cuda` is not present or points to another CUDA version then this may fail, and you will need to specify `CUDA_HOME` directly or fix the symlink.

The HiPIMS library is created as follows. `HIPIMS_ROOT` is used here to indicate the root of the repository, it is not necessary to actually set this variable.
```
cd $HIMPS_ROOT/cuda/
python setup.py install
```

### Running Locally
The DAFNI HiPIMS application is not designed to be run locally.

### Running Tests
There is a test python script which runs a single GPU example case. Firstly, download the test data from DAFNI, as outlined in `data/inputs/README.md`. Then the test application can be run using
```
cd $HIPIMS_ROOT
python singleGPU_example.py
```
The results of the simulation should appear in the data/outputs directory.

### Testing on an Azure VM
#### _Creating the Virtual Machine_
If you have no local GPU hardware to test the application on, then an Azure GPU-enabled Virtual Machine (VM) can be created using Terraform. Note that the Terraform configuration files use the NvidiaGpuDriverLinux extension to install the NVidia GPU libraries, so you won't need to do this manually once the VM is up and running.
```
cd tf
terraform plan
terraform apply
```

Once finished testing, remove the VM using
```
terraform destroy
```
*It is extremely important that you remove the VM once testing is complete as running an Azure VM is very expensive!*

On creation of the Azure VM, an external IP address and key pair are created. The IP address is displayed by Terraform on completion of the `apply` procedure, and it can also be found through the Azure portal - make a copy of this onto the clipboard. The private key is stored in Terraform state and is not displayed on the terminal. To recover the key, first make sure that `jq` is installed:
```
sudo apt install jq
```
Next, [recover the VM private key from the terraform state](https://devops.stackexchange.com/questions/14493/get-private-key-from-terraform-state) using the following shell commands:
```
mkdir ~/.ssh
terraform show -json | \
jq -r '.values.root_module.resources[].values | select(.private_key_pem) |.private_key_pem' \
> ~/.ssh/pyramidvm_private_key.pem
chmod og-rw ~/.ssh/pyramidvm_private_key.pem
```
Then you can log into the VM, which has the hostname `pyramidvm` using
```
ssh -i ~/.ssh/pyramidvm_private_key.pem <ip address> -l pyramidtestuser
```
This private key is usable until the VM is destroyed - the key will need to be recovered again for each time that terraform is used to recreate the VM in Azure.

#### _Setting up Docker on the Virtual Machine_
By default, Azure VMs are supplied with an attached temporary disk under `/mnt'. This should be sufficient for application testing purposes as the intention with the VM is to create the VM, clone and build the repo, test the application, and delete the VM. The Terraform file for the VM does contain some commented-out configuration for adding another disk to the Azure resource group that can be mounted to the VM filesystem, if this is needed.

[Docker](https://www.docker.com/) will need to be installed on the VM, and also the default directory under which it stores images needs to be moved to the temporary disk under `/mnt`, because the OS disk is not large enough to hold the HiPIMS docker images. Note that all Docker commands will need to be run under `sudo`.

Firstly, [install Docker](https://docs.docker.com/engine/install/ubuntu/):

```
cd
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt-get install ca-certificates curl gnupg lsb-release
```
Add Docker's official GPG key:
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```
This command sets up the stable repository:
```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
Finally install the Docker Engine and check that it is working correctly:
```
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
sudo docker run hello-world
```

Moving the Docker images directory involves a bit of work.Rather than attempting to entirely move the default location, which involves changing Docker's configuration files, opting for a [bind mount solution](https://www.ibm.com/docs/en/cloud-private/3.1.0?topic=pyci-specifying-default-docker-storage-directory-by-using-bind-mount) is a bit less painless. First, remove any running containers and images:
```
sudo docker rm -f $(sudo docker ps -aq)
sudo docker rmi -f $(sudo docker images -q)
```
Stop the Docker service and remove the Docker storage directory
```
sudo systemctl stop docker
sudo rm -rf /var/lib/docker
```
Create a new Docker storage directory and bind it to a directory on `/mnt`:
```
sudo mkdir /var/lib/docker
sudo mkdir /mnt/docker
sudo mount --rbind /mnt/docker /var/lib/docker
```
Restart the Docker service:
```
sudo systemctl start docker
```
Alternatively - Create the bind mount point first, and then install Docker!

Finally, Docker needs to be prepared for [using the GPU hardware](https://www.cloudsavvyit.com/14942/how-to-use-an-nvidia-gpu-with-docker-containers/). Check that the NVidia drivers are actually installed using `nvidia-smi`:
```
pyramidtestuser@pyramidtestvm:/mnt/PYRAMID-HiPIMS$ nvidia-smi
Wed Apr 13 12:48:49 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000001:00:00.0 Off |                  Off |
| N/A   30C    P0    24W / 250W |      0MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```
We then need to add the NVidia Container Toolkit for Docker, which integrates into the Docker Engine to provide GPU support.
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```
At this point we can test the Docker NVidia integration to make sure it is working:
```
pyramidtestuser@pyramidtestvm:/mnt/PYRAMID-HiPIMS$ sudo docker run -it --gpus all nvidia/cuda:11.4.0-base-ubuntu20.04 nvidia-smi
Wed Apr 13 12:53:40 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000001:00:00.0 Off |                  Off |
| N/A   30C    P0    24W / 250W |      0MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```
The Docker and NVidia integration should now be ready to run the application.

#### _Cloning the application repository and building the app_
Git should be installed on the Virtual Machine, and the next thing to do is to clone the HiPIMS repository for building the application. Firstly you will need to enable access to the repository. The easiest way is probably to create a Personal Access Token (PAT) in GitHub, by following the [GitHub documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token). Having created the PAT you will then be able to clone the repo. When running `sudo git clone` you will enter your GitHub username but then instead of a password, paste in the PAT from GitHub. Note that the PAT is a one-time only generation and you will not be able to see it again once you navigate away from the PAT page, so make sure you copy it to clone the repo or you will end up having to regenerate it.
```
cd /mnt
sudo git clone https://github.com/NCL-PYRAMID/PYRAMID-HiPIMS.git
```

Finally we can build the HiPIMS Docker container.
```
cd PYRAMID-HiPIMS
sudo docker build . -t pyramid-hipims
```

## Deployment
### Local
Build the Docker container for the HiPIMS application using a standard `docker build` command.
```
docker build . -t pyramid-hipims
```
The container is designed to mount a local directory for reading and writing. For testing locally, download the test data from DAFNI, as outlined in `data/inputs/README.md`. Then the test application can be run using
```
docker run -v "$(pwd)/data:/data" pyramid-hipims
```
Data produced by the application will be in data/outputs. Note that because of the way that Docker permissions work, these data will have `root/root` ownership permissions. You will need to use elevated `sudo` privileges to delete the outputs folder.

### Production
#### DAFNI upload
The model is containerised using Docker, and the image is _tar_'ed and _zip_'ed for uploading to DAFNI. Use the following commands in a *nix shell to accomplish this.

```
docker build . -t pyramid-hipims
docker save -o pyramid-hipims.tar pyramid-hipims:latest
gzip pyramid-hipims.tar
```

The `pyramid-hipims.tar.gz` Docker image and accompanying DAFNI model definintion file (`model-definition.yml`) can be uploaded as a new model using the "Add model" facility at [https://facility.secure.dafni.rl.ac.uk/models/](https://facility.secure.dafni.rl.ac.uk/models/).

## Usage
The deployed model can be run in a DAFNI workflow. See the [DAFNI workflow documentation](https://docs.secure.dafni.rl.ac.uk/docs/how-to/how-to-create-a-workflow) for details.

When running the model as part of a larger workflow receiving data from the PYRAMID deep learning component, all data supplied to the model will appear in the folder `/data/inputs`, exactly as produced by the deep learning model. The outputs of this converter must be written to the `/data/outputs` folder within the Docker container. When testing locally, these paths are instead `./data/inputs' and './data/outputs' respectively. The Python script is able to determine which directory to use by checking the environment variable `PLATFORM`, which is set in the Dockerfile.

Model outputs in `/data/outputs` should contain the actual data produced by the converter, as well as a `metadata.json` file which is used by DAFNI in publishing steps when creating a new dataset from the data produced by the model.

## Roadmap
- [x] Initial Research  
- [ ] Minimum viable product <-- You are Here  
- [ ] Alpha Release  
- [ ] Feature-Complete Release  

## Contributing
TBD

### Main Branch
All development can take place on the main branch. 

## License
This repository is private to the PYRAMID project.

The official version of HiPIMS is maintained by the Hydro-Environmental Modelling Labarotory at Loughborough University. Please contact Qiuhua Liang (q.liang@lboro.ac.uk) for more information.

## Acknowledgements
This work was funded by NERC, grant ref. NE/V00378X/1, “PYRAMID: Platform for dYnamic, hyper-resolution, near-real time flood Risk AssessMent Integrating repurposed and novel Data sources”. See the project funding [URL](https://gtr.ukri.org/projects?ref=NE/V00378X/1).

