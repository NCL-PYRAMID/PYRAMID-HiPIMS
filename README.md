# PYRAMID-HiPIMS
CUDA-enabled HiPIMS for DAFNI

## About

### Project Team
Xue Tong, Loughborough University ([x.tong2@lboro.ac.uk](mailto:x.tong2@lboro.ac.uk))  
Elizabeth Lewis, Newcastle University  ([elizabeth.lewis2@newcastle.ac.uk](mailto:elizabeth.lewis2@newcastle.ac.uk))  

### RSE Contact
Robin Wardle  
RSE Team, Newcastle Data  
Newcastle University NE1 7RU  
([robin.wardle@newcastle.ac.uk](mailto:robin.wardle@newcastle.ac.uk))  

## Built With


## Getting Started

### Prerequisites


### Installation


### Running Locally


### Running Tests

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
This code is private to the PYRAMID project.

## Acknowledgements
This work was funded by NERC, grant ref. NE/V00378X/1, “PYRAMID: Platform for dYnamic, hyper-resolution, near-real time flood Risk AssessMent Integrating repurposed and novel Data sources”. See the project funding [URL](https://gtr.ukri.org/projects?ref=NE/V00378X/1).

