# conformal-safety-learning
Companion code for the paper Learning Robot Safety from Sparse Human Feedback using Conformal Prediction.

Please reach out to aofeldma@stanford.edu for any code issues/questions.

Note: the following procedure was tested using a WSL2 linux environment on a computer with NVIDIA GPU

## Installation and Setup

### Install the package
    a. Navigate to conformal-safety-learning
    b. Make a new conda environment: conda create --name cp-safety -y python=3.8
    c. Activate the environment: conda activate cp-safety
    d. Upgrade pip: python -m pip install --upgrade pip
    e. Separately install nerfstudio, following https://docs.nerf.studio/quickstart/installation.html
    (This can be skipped if you do not want to use nerf functionality)
    f. Install other dependencies: pip install -e .

### Point Gaussian splat to correct directory (assuming nerfstudio installed)
    a. Open config.yml found at data\nerf_data\outputs\cp_1203_0\splatfacto\2024-12-03_150258
    b. Under the line "data: &id003 !!python/object/apply:pathlib.PosixPath", modify the path to your absolute path to data/nerf_data/cp_1203_0, with a new hyphenated line for each subdirectory
    c. Under the line "output_dir: !!python/object/apply:pathlib.PosixPath", similarly modify with the absolute path to data/nerf_data/outputs

## Scripts Organization
    a. Basic Tools: Contains scripts for basic functionality like simulation experiment setup, visualization, etc.
    b. Conformal: Contains scripts for basic functionality like simulation experiment setup, visualization, etc.
    c. Experiments: Contains scripts for generating the experimental results and figures
    d. Policies: Contains scripts for control of drone
    e. PolicyModification: Contains scripts for policy modification approaches
    f. Transformers: Contains scripts for different representation learning approaches
    g. Warning System: Contains scripts for warning system functions and subdirectory for baselines

## Example Use
### Note: navigate to conformal-safety-learning/Scripts before running
    a. Generate an experiment setup: python -m BasicTools.experiment_info with SAVE = True
    b. Generate original policy: under Policies, run scp_mpc.py with SAVE=True
    c. Generate warning system: under Experiments, run warning_demo.py
    d. Generate modified policy: under Experiments, run backup_demo.py