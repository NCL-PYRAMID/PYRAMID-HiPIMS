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
These frameworks require PyTorch 1.2 or higher. The dependent libs can be found in the requirements.txt. Specifically, it needs:

Linux
Python 3.7
PyTorch 1.2
CUDA 10.0
GCC 4.9+

### Installation
a. Creat and activate an anaconda virtual environment 

conda create --name hipims python=3.7
conda activate hipims

b. Install PyTorch stable or nightly and torchvision following the official instructions.

conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

c. install the necessary packages

conda install numpy matplotlib pyqt seaborn tqdm kiwisolver
conda install rasterio pysal pyproj rasterstats geopy cartopy contextily earthpy folium 
conda install geojson mapboxgl hydrofunctions geocoder tweepy

d. build the lib

cd [path]/cuda/
python setup.py install

### Running Tests

cd ..
python singleGPU_example.py

## Deployment

### Local
A local Docker container that mounts the test data can be built using:

```
docker build . -t pyramid-hipims
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

