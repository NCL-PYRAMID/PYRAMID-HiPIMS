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

#### GCC / G++ versions
CUDA 10.2 Toolkit requires GCC and G++ for its installation. GCC versions later than 8 are not supported by CUDA 10.2. This deployment uses GCC 7. Under Linux it is possible to select compiler versions using `update-alternatives`. To install GCC 7 as the default compiler under Linux, install and select it with the following commands:
```
sudo apd update
sudo apt install gcc-7 g++-7
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
- [The latest version]
- [All archives](https://developer.nvidia.com/cuda-toolkit-archive)
- [CUDA 10.2 archive](https://developer.nvidia.com/cuda-10.2-download-archive)

If your build machine does not already have CUDA installed or if it is older than version 10.2, then navigate to the [CUDA 10.2 archive](https://developer.nvidia.com/cuda-10.2-download-archive) and download the appropriate **runfile** for your system. Navigate to an appropriate directory and download the installer file and two patches using `wget` in a terminal window.
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
Alternatively, the following removal steps might be useful:
```
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" 
sudo apt-get --purge remove "*nvidia*"
sudo rm -rf /usr/local/cuda*
```



#### Anaconda installation
HiPIMS is built around PyTorch 1.2 or higher. The dependent packages can be found in the requirements.txt. Specifically, it needs:

Linux  
Python 3.7  
PyTorch 1.2  
CUDA 10.0  
GCC 4.9+  


a. Install Anaconda
```
conda update -n base -c defaults conda
```

b. Initialise Anaconda to use your preferred login shell
```
conda init bash
```
You will need to log out of this shell and re-log in to ensure that Anaconda works correctly with the shell

#### Anaconda environment configuration
b. Create and activate an Anaconda virtual environment with Python 3.7. Some of the packages used in the application are only available from `conda-forge`, so all packages will use `conda-forge` as the default installation channel.

```
conda create -c conda-forge -y --name hipims python=3.7
conda activate hipims
```

b. Install pytorch, torchvision and the CUDA toolkit.

```
conda install -c conda-forge -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
```

c. Install the python packages used by HiPIMS.

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

#### Alternative Anaconda environment
Alternatively, install from an environment file:
```
conda env create -f hipims-environment.yml -y
conda activate hipims
```


### Installation


d. build the lib

set CUDA_HOME

cd [path]/cuda/
python setup.py install

### Running Locally


### Running Tests

cd ..
python singleGPU_example.py

## Deployment

### Local
A local Docker container that mounts the test data can be built using:

```
docker build . -t pyramid-hipims
```

MISSING
```
NewcastleCivilCentre/output
```

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

