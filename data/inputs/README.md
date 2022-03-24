# HiPIMS Input Data
This directory contains the input data for HiPIMS.

## Running the application within the operating system on a workstation
The application will read from the
```
data/inputs
```
directory, and write to the
```
data/outputs
```
directory.

## Running the application within a Docker container
The application will read from the
```
/data/inputs
```
directory, and write to the
```
/data/outputs
```
directory. Note the root filesystem slash. The `/data` directory in the Docker container should be mapped to the `$HIPIMS_ROOT/data` directory. The test data should *not* be copied to this directory in the Docker container.

## Running the application on HiPIMS
As above, the application will read from the
```
/data/inputs
```
directory, and write to the
```
/data/outputs
```
directory. Note the root filesystem slash. The data for analysis will come from a prior model in the PYRAMID workflow, and hence any test data should *not* be included in this directory in the Docker container.

## Test Data
A test data set is present on DAFNI. This data set is identified as:
- Title: PYRAMID HiPIMS Test Data Set
- Dataset ID: 1fa68701-0a4a-4aa7-8e8e-e3f7707c93d6

To use this data when workstation testing, download it from DAFNI into the `$HIPIMS_ROOT/data/inputs` directory. It does not need to be downloaded when testing on DAFNI as the application is set up to read from this dataset within the DAFNI platform. The data is in the form of a zip file - it should be unzipped directly into the data/inputs directory, i.e. all files should be in this directory.
