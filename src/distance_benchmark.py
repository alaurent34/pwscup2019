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
from joblib import Parallel, delayed
import traj_dist.distance as tdist

PARAMS = {
    "dist" : ["sspd", "discret_frechet", "hausdorff", "dtw", "lcss", "edr", "erp"] # frechet
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

    res_d1 = (np.count_nonzero(cdist_matrix_d1.diagonal() == min_cdist_m_d1)/
              min_cdist_m_d1.shape[0])
    res_d2 = (np.count_nonzero(cdist_matrix_d2.diagonal() == min_cdist_m_d2)/
              min_cdist_m_d2.shape[0])
    res_d1_d2 = (np.count_nonzero(cdist_matrix_d1_d2.diagonal() == min_cdist_m_d1_d2)
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
    # transform to boolean array
    assert ((org_d1.shape[0] == org_d2.shape[0] and ref_d1.shape[0] == ref_d2.shape[0])
            and ref_d1.shape[0] == org_d1.shape[0])

    upper_bound = max(org_d1.max(), org_d2.max(), ref_d1.max(), ref_d2.max())
    shape_b = (org_d1.shape[0], upper_bound+1)
    org_d1_b = np.zeros(shape_b)
    org_d2_b = np.zeros(shape_b)
    ref_d1_b = np.zeros(shape_b)
    ref_d2_b = np.zeros(shape_b)

    for i in range(shape_b[0]):
        org_d1_b[i, org_d1[i]] = 1
        org_d2_b[i, org_d2[i]] = 1
        ref_d1_b[i, ref_d1[i]] = 1
        ref_d2_b[i, ref_d2[i]] = 1

    # compute pairwise distance
    cdist_matrix_d1 = cdist(org_d1, ref_d1, metric="jaccard")
    cdist_matrix_d2 = cdist(org_d2, ref_d2, metric="jaccard")
    cdist_matrix_d1_d2 = cdist_matrix_d1 + cdist_matrix_d2

    # compare minimum with diagonal of matrix
    min_cdist_m_d1 = cdist_matrix_d1.min(axis=1)
    min_cdist_m_d2 = cdist_matrix_d2.min(axis=1)
    min_cdist_m_d1_d2 = (cdist_matrix_d1_d2).min(axis=1)

    res_d1 = (np.count_nonzero(cdist_matrix_d1.diagonal() == min_cdist_m_d1)/
              min_cdist_m_d1.shape[0])
    res_d2 = (np.count_nonzero(cdist_matrix_d2.diagonal() == min_cdist_m_d2)/
              min_cdist_m_d2.shape[0])
    res_d1_d2 = (np.count_nonzero(cdist_matrix_d1_d2.diagonal() == min_cdist_m_d1_d2)
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
    logger = verboselogs.VerboseLogger("benchLogger")
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

    logger.info("Benchmark is started !")
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
    logger.debug("Read numpy arrays (traj)")

    # Computing scores
    results = (Parallel(n_jobs=-1)(delayed(tdist_cdist_wrapper)(
        metric, org_latlng_traj_d1, org_latlng_traj_d2,
        ref_latlng_traj_d1, ref_latlng_traj_d2
        ) for metric in PARAMS["dist"]))
    logger.debug("Lat, Lng parrallell computing done")

    results.append(jaccard(org_cell_traj_d1, org_cell_traj_d2,
                           ref_cell_traj_d1, ref_cell_traj_d2))
    logger.debug("Cells computing done")

    # list to dictionary
    for i in range(1, len(results)):
        results[0].update(results[i])
    results = results[0]
    logger.debug("transform into dic done")
    logger.info("Computation over")

    # saving results
    df_res = pd.DataFrame(results).T
    org_file = args.traj_org.split("/")[-1]
    ref_file = args.traj_ref.split("/")[-1]
    os.makedirs(f"../data/output/{org_file}_VS_{ref_file}/", exist_ok=True)
    df_res.to_csv(f"../data/output/{org_file}_VS_{ref_file}/results.csv")
    logger.info("Saving over")

if __name__ == "__main__":
    main()
