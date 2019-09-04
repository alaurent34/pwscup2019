"""
File: distance_benchmark.py
Author: Antoine Laurent
Email: laurent.antoine@courrier.uqam.ca
Github: https://github.com/Drayer34
Description: Benchmark of privacy with multiple distance
"""

import os
import argparse
import logging
import verboselogs

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.externals.joblib import Parallel, delayed
import traj_dist.distance as tdist

PARAMS = {
    "dist" : ["sspd", "frechet", "discret_frechet", "hausdorff", "dtw", "lcss", "edr", "erp"]
        }

def tdist_cdist_wrapper(metric, org_d1, org_d2, ref_d1, ref_d2):
    """TODO: Docstring for tdist_cdist_wrapper.
    :returns: TODO

    """
    # compute pairwise distance
    if not (metric in ["frechet", "discret_frechet"]):
        cdist_matrix_d1 = tdist.cdist(org_d1, ref_d1, metric=metric, type_d="spherical")
        cdist_matrix_d2 = tdist.cdist(org_d2, ref_d2, metric=metric, type_d="spherical")
    else:
        cdist_matrix_d1 = tdist.cdist(org_d1, ref_d1, metric=metric, type_d="euclidean")
        cdist_matrix_d2 = tdist.cdist(org_d2, ref_d2, metric=metric, type_d="euclidean")

    cdist_matrix_d1_d2 = cdist_matrix_d1 + cdist_matrix_d2

    # compare minimum with diagonal of matrix
    min_cdist_m_d1 = cdist_matrix_d1.min(axis=1)
    min_cdist_m_d2 = cdist_matrix_d2.min(axis=1)
    min_cdist_m_d1_d2 = (cdist_matrix_d1_d2).min(axis=1)

    res_d1 = (np.count_nonzero(cdist_matrix_d1.diagonal == min_cdist_m_d1)/min_cdist_m_d1.shape[0])
    res_d2 = (np.count_nonzero(cdist_matrix_d2.diagonal == min_cdist_m_d2)/min_cdist_m_d2.shape[0])
    res_d1_d2 = (np.count_nonzero(cdist_matrix_d1_d2.diagonal == min_cdist_m_d1_d2)
                 /min_cdist_m_d1_d2.shape[0])

    return {metric: {
                "d1": res_d1,
                "d2": res_d2,
                "d1_d2": res_d1_d2
                }
            }

def jaccard_distance(arr_a, arr_b):
    """
    Compute Jaccard distance
    """
    set_a = set(arr_a)
    set_b = set(arr_b)
    return 1 - (len(set_a.intersection(set_b))/len(set_a.union(set_b)))

def jaccard(org_d1, org_d2, ref_d1, ref_d2):
    """TODO: Docstring for jaccard.
    :returns: TODO

    """
    # compute pairwise distance
    cdist_matrix_d1 = cdist(org_d1, ref_d1, metric=jaccard_distance)
    cdist_matrix_d2 = cdist(org_d2, ref_d2, metric=jaccard_distance)
    cdist_matrix_d1_d2 = cdist_matrix_d1 + cdist_matrix_d2

    # compare minimum with diagonal of matrix
    min_cdist_m_d1 = cdist_matrix_d1.min(axis=1)
    min_cdist_m_d2 = cdist_matrix_d2.min(axis=1)
    min_cdist_m_d1_d2 = (cdist_matrix_d1_d2).min(axis=1)

    res_d1 = (np.count_nonzero(cdist_matrix_d1.diagonal == min_cdist_m_d1)/min_cdist_m_d1.shape[0])
    res_d2 = (np.count_nonzero(cdist_matrix_d2.diagonal == min_cdist_m_d2)/min_cdist_m_d2.shape[0])
    res_d1_d2 = (np.count_nonzero(cdist_matrix_d1_d2.diagonal == min_cdist_m_d1_d2)
                 /min_cdist_m_d1_d2.shape[0])

    return {"jaccard": {
                "d1": res_d1,
                "d2": res_d2,
                "d1_d2": res_d1_d2
                }
            }

def main():
    """ Run preprocessing
    """
    # init logger
    logger = verboselogs.VerboseLogger("preproLogger")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # init parser
    parser = argparse.ArgumentParser("This file preprocess data from PWSCup2019")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-o", "--traj_org", help="Path of the orginal traj directory", type=str,
                        required=True)
    parser.add_argument("-r", "--traj_ref", help="Path of the reference traj directory",
                        type=str, required=True)

    # recover args
    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode on")
        logger.setLevel(logging.DEBUG)

    # read data
    ## original data set
    # org_df = pd.read_csv(f"{args.traj_org}/dataframe.csv")
    org_cell_traj_d1 = np.load(f"{args.traj_org}/cell_traj_d1.npy")
    org_cell_traj_d2 = np.load(f"{args.traj_org}/cell_traj_d2.npy")
    org_latlng_traj_d1 = np.load(f"{args.traj_org}/lat_lng_traj_d1.npy")
    org_latlng_traj_d2 = np.load(f"{args.traj_org}/lat_lng_traj_d2.npy")

    ## reference data set
    # ref_df = pd.read_csv(f"{args.traj_ref}/dataframe.csv")
    ref_cell_traj_d1 = np.load(f"{args.traj_ref}/cell_traj_d1.npy")
    ref_cell_traj_d2 = np.load(f"{args.traj_ref}/cell_traj_d2.npy")
    ref_latlng_traj_d1 = np.load(f"{args.traj_ref}/lat_lng_traj_d1.npy")
    ref_latlng_traj_d2 = np.load(f"{args.traj_ref}/lat_lng_traj_d2.npy")

    # Computing scores
    results = (Parallel(n_jobs=-1)(delayed(tdist_cdist_wrapper)(
        metric, org_latlng_traj_d1, org_latlng_traj_d2,
        ref_latlng_traj_d1, ref_latlng_traj_d2
        ) for metric in PARAMS["dist"]))

    results.append(jaccard(org_cell_traj_d1, org_cell_traj_d2,
                           ref_cell_traj_d1, ref_cell_traj_d2))

    # list to dictionary
    for i in range(1, len(results)):
        results[0].update(results[i])
    results = results[0]

    # saving results
    df_res = pd.DataFrame(results).T
    org_file = args.traj_org.split("/")[-1]
    ref_file = args.traj_ref.split("/")[-1]
    os.makedirs(f"../data/output/{org_file}_VS_{ref_file}/", exist_ok=True)
    df_res.to_csv(f"../data/output/{org_file}_VS_{ref_file}/results.csv")

if __name__ == "__main__":
    main()
