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
1(bis). You can also download testing data [here](https://cloud.ikb.info.uqam.ca/index.php/s/T3FmdLRmbi2C9Bk)

2. Preprocessing of the data with `preprocessing.py`
```shell
preprocessing.py -i <your_trajectory_input_file> -o <name_of_directory_for_output> -r <information_region_file>
```

3. Runing benchmark (you need both original and reference files). An example
   using the testings files

```shell
python distance_benchmark.py \
    -o ../data/testing_data/output_org_osaka \
    -r ../data/testing_data/output_ref_osaka
```

### Re-identification

```shell
cd data/reid_preliminary_phase/
./prepro.sh /PubData
./prepro.sh /RefData_000-021
```

```shell
cd src/
./attack.sh ../data/reid_preliminary_phase/prepro
```
