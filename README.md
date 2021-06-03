#Flight Track Prediction
This repo includes the coding and results involved in creating a deep learning model for 4D (Latitude, Longitude, Altitude, Time) flight trajectory prediction.

## Using the Repo
Data should be processed as interpolated sequences of 4D flight plans, 4D trajectory labels, and collections of surrounding weather cubes (assumed *20x20xZ*, where *Z* is an odd integer). These may be processed using [Weather-Preprocessing](https://github.com/schimpfen/Weather-Preprocessing).

Project was developed and tested using PyCharm 2020.1 on Ubuntu 20.04 LTS. If using PyCharm, it is strongly encouraged to disable indexing of the `data`, `Initialized Plots`, and `Output` directories.

* To disable indexing, right click on Data in the project explorer, and follow mark as | excluded

This repo is under continuous development. As such, see release tags for project status associated with any papers or presentations.
### Environment
It is recommended to build the provided `environment.yml` in Anaconda. Otherwise, project libraries include:
* basemap 
* CUDAToolkit 10.2
* colorama
* cupy-cuda102
* matplotlib
* netcdf4
* numpy
* pynvrtc
* python 3.8
* scikit-learn
* torchaudio
* torchvision
* tqdm
* ray[tune]

# Uses
The status of the project in this commit provides models and results discussed
> *Flight Trajectory Prediction Based on Hybrid-Recurrent Networks*\
> N. Schimpf, E. Knoblock, Z. Wang, R. Apaza, H. Li\
> Cognitive Communications and Aeronautical Applications Workshop (CCAA 2021)
* `train.py` provided a method for training mode and reporting their training error
* `Model Eval.py` runs an execution of the determined test datasets for each saved model, storing the results in `Output`
* `Inverse Normalize.py` provides a method for scaling model predictions, flight plans, and labelled trajectories to actual latitude and longitude coordinates. The scaling values may need adjustment depending on the flight data preprocessed.
* `Reporting.py` Generates visual reports and statistics for each de-normalized set of predictions. Statistics include:
  * Pointwise Horizontal Error (Mean Absolute and Standard Deviation)
  * Pointwise Vertical Error (Mean Absolute and Standard Deviation)
  * Trajectorywise Horizontal Error (Mean Absolute and Standard Deviation)
  * Trajectorywise Vertical Error (Mean Absolute and Standard Deviation)