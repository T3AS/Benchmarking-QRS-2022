[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This is a repository for the paper "Evaluating the Robustness of Deep Reinforcement Learning for Autonomous Policies in a Multi-agent Urban Driving Environment"

### Installattion Setup

1. Install the system requirements:
    - Ubuntu 18.04+ 
    - Anaconda (latest version)
	- cmake (`sudo apt install cmake`)
	- zlib (`sudo apt install zlib1g-dev`)
	- [optional] ffmpeg (`sudo apt install ffmpeg`)

2. Setup CARLA (0.9.4):

Run `mkdir ~/software && cd ~/software`

Download the 0.9.4 release version from: [Here](https://drive.google.com/file/d/1p5qdXU4hVS2k5BOYSlEm7v7_ez3Et9bP/view)
Extract it into `~/software/CARLA_0.9.4`
    
Run `echo "export CARLA_SERVER=${HOME}/software/CARLA_0.9.4/CarlaUE4.sh" >> ~/.bashrc`

3. Install the libraries:

Fork/Clone the repository to your workspace:

Create a new conda env named "Benchmarking" and install the required packages:
`conda env create -f conda_env.yml`

Activate the environment:
`conda activate macad-gym-benchmarking`

Run the following commands in sequence for installing rest of the packages to avoid version errors:

`pip install -e .`

`pip install --upgrade pip`

`pip install -e .` 

`pip install tensorflow==2.1.0`

`pip install tensorflow-gpu==2.1.0`


`pip install pip install ray[tune]==0.8.4`

`pip install pip install ray[rllib]==0.8.4`


`pip install tf_slim`

`pip install tensorboardX==2.1`

### Instructions
Soon to be updated here along with the paper.


