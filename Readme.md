# PWSCup 2019 - Team 11 : Unknown IPA

This repesotory contains all script to do preprocessing, anonymisation an
attackinf trajectories dataset from PWSCup contest

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install
requirements of this repo.

```bash
pip install -r requirements.txt
```

Note that traj_dist package must be installed [here](https://github.com/djjavo/traj-dist)

## Usage

1. Download PWSCup data from the team Google Drive account
1. (bis) You can also download testing data [here](https://www.iwsec.org/pws/2019/data/PubInfo_20190828.zip)
2. Preprocessing of the data with `preprocessing.py`

```shell
preprocessing.py -i <your_trajectory_input_file> -o <name_of_directory_for_output> -r <information_region_file>
```
