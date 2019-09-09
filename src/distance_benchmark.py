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
        "dist": ["sspd", "discret_frechet", "hausdorff", "dtw",
                 "lcss", "edr", "erp", "jaccard", "pfipf"],

        "lat_lng_dist": ["sspd", "discret_frechet", "hausdorff",
                         "dtw", "lcss", "edr", "erp"], # frechet

        "cell_dist": ["jaccard", "pfipf"],

        "run": "pfipf",

        "alpha": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
                  "1", "2", "3", "4", "round"]
        }

def evaluate(cdist_matrix_d1, cdist_matrix_d2, cdist_matrix_d1_d2=np.empty(0)):
    """TODO: Docstring for evaluate.
    :returns: TODO

    """
    if cdist_matrix_d1_d2.shape[0] == 0:
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

    return res_d1, res_d2, res_d1_d2

def cdist_wrapper(metric, path, l_org_d1, l_org_d2, l_ref_d1, l_ref_d2,
                  c_org_d1, c_org_d2, c_ref_d1, c_ref_d2):
    """
    doc
    """
    if metric in PARAMS["lat_lng_dist"]:
        return latlng_cdist_wrapper(metric, path, l_org_d1, l_org_d2, l_ref_d1, l_ref_d2)

    return cell_cdist_wrapper(metric, path, c_org_d1, c_org_d2, c_ref_d1, c_ref_d2)

def cell_cdist_wrapper(metric, path, org_d1, org_d2, ref_d1, ref_d2, alpha=0):
    """TODO: Docstring for function.

    :arg1: TODO
    :returns: TODO

    """
    # preprocess
    assert ((org_d1.shape == org_d2.shape and ref_d1.shape == ref_d2.shape)
            and ref_d1.shape == org_d1.shape)

    upper_bound = max(org_d1.max(), org_d2.max(), ref_d1.max(), ref_d2.max())

    if metric == "jaccard":
        org_d1d2 = np.concatenate([org_d1, org_d2], axis=1)
        org_d1d2 = transform_jaccard(org_d1d2, upper_bound, org_d1.shape[0])
        org_d1 = transform_jaccard(org_d1, upper_bound, org_d1.shape[0])
        org_d2 = transform_jaccard(org_d2, upper_bound, org_d1.shape[0])
        ref_d1d2 = np.concatenate([ref_d1, ref_d2], axis=1)
        ref_d1d2 = transform_jaccard(ref_d1d2, upper_bound, org_d1.shape[0])
        ref_d1 = transform_jaccard(ref_d1, upper_bound, org_d1.shape[0])
        ref_d2 = transform_jaccard(ref_d2, upper_bound, org_d1.shape[0])
        distance = "jaccard"
    else:
        org_d1d2 = np.concatenate([org_d1, org_d2], axis=1)
        org_d1d2 = compute_pfipf(org_d1d2, upper_bound, org_d1.shape[0])
        org_d1 = compute_pfipf(org_d1, upper_bound, org_d1.shape[0])
        org_d2 = compute_pfipf(org_d2, upper_bound, org_d1.shape[0])
        ref_d1d2 = np.concatenate([ref_d1, ref_d2], axis=1)
        ref_d1d2 = compute_pfipf(ref_d1d2, upper_bound, org_d1.shape[0])
        ref_d1 = compute_pfipf(ref_d1, upper_bound, org_d1.shape[0])
        ref_d2 = compute_pfipf(ref_d2, upper_bound, org_d1.shape[0])
        distance = "cosine"

    # troncature with alpha
    if alpha == "round":
        org_d1d2 = np.round(org_d1d2)
        ref_d1d2 = np.round(ref_d1d2)
    else:
        alpha = float(alpha)
        org_d1d2[np.nonzero(org_d1d2 > alpha)] = 1
        # org_d1d2[np.nonzero(org_d1d2 < alpha)] = 0
        ref_d1d2[np.nonzero(ref_d1d2 > alpha)] = 1
        # ref_d1d2[np.nonzero(ref_d1d2 < alpha)] = 0

    # computing pairwise distance
    cdist_matrix_d1 = cdist(org_d1, ref_d1, metric=distance)
    cdist_matrix_d2 = cdist(org_d2, ref_d2, metric=distance)
    cdist_matrix_d1_d2 = cdist(org_d1d2, ref_d1d2, metric=distance)


    # saving
    np.save(f"{path}/{metric}_cdist_matrix_d1.npy", cdist_matrix_d1)
    np.save(f"{path}/{metric}_cdist_matrix_d2.npy", cdist_matrix_d2)
    np.save(f"{path}/{metric}_cdist_matrix_d1d2_alpha={alpha}.npy", cdist_matrix_d1_d2)

    res_d1, res_d2, res_d1_d2 = evaluate(cdist_matrix_d1, cdist_matrix_d2, cdist_matrix_d1_d2)

    return {f"{metric}_alpha={alpha}": {
                "d1": res_d1,
                "d2": res_d2,
                "d1_d2": res_d1_d2
                }
            }

def latlng_cdist_wrapper(metric, path, org_d1, org_d2, ref_d1, ref_d2):
    """TODO: Docstring for latlng_cdist_wrapper.
    :returns: TODO

    """
    # compute pairwise distance
    if not (metric in ["frechet", "discret_frechet"]):
        cdist_matrix_d1 = tdist.cdist(org_d1, ref_d1, metric=metric, type_d="spherical")
        cdist_matrix_d2 = tdist.cdist(org_d2, ref_d2, metric=metric, type_d="spherical")
    else:
        cdist_matrix_d1 = tdist.cdist(org_d1, ref_d1, metric=metric, type_d="euclidean")
        cdist_matrix_d2 = tdist.cdist(org_d2, ref_d2, metric=metric, type_d="euclidean")

    # saving
    np.save(f"{path}/{metric}_cdist_matrix_d1.npy", cdist_matrix_d1)
    np.save(f"{path}/{metric}_cdist_matrix_d2.npy", cdist_matrix_d2)

    res_d1, res_d2, res_d1_d2 = evaluate(cdist_matrix_d1, cdist_matrix_d2)

    return {metric: {
                "d1": res_d1,
                "d2": res_d2,
                "d1_d2": res_d1_d2
                }
            }

def compute_pfipf(cell_traj, upper_bound, number_people):
    """TODO: Docstring for compute_pfipf.
    :returns: TODO

    """
    # computing pf matrix
    pf_ = np.zeros((number_people, upper_bound+1))
    for i in range(number_people):
        index, count = np.unique(cell_traj[i], return_counts=True)
        pf_[i, index] = count/cell_traj.shape[1]

    ipf_ = np.zeros(upper_bound+1)
    for i in range(upper_bound+1):
        ipf_[i] = np.count_nonzero(pf_[:, i])
    ipf_ = np.log(number_people/(ipf_+1)) # to avoid division by 0

    return pf_ * ipf_

def transform_jaccard(cell_traj, upper_bound, nb_ligne):
    """TODO: Docstring for transform.

    :arg1: TODO
    :returns: TODO

    """
    arr_b = np.zeros((nb_ligne, upper_bound+1))
    for i in range(nb_ligne):
        arr_b[i, cell_traj[i]] = 1

    return arr_b

def main():
    """ Run distane benchmark
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
    parser.add_argument("-d", "--distance", help="Distance wich will be computed", default=None,
                        type=str)

    # recover args
    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode on")
        logger.setLevel(logging.DEBUG)
    if args.distance:
        PARAMS["dist"] = []
        PARAMS["dist"].append(args.distance)

    org_file = args.traj_org.split("/")[-1]
    ref_file = args.traj_ref.split("/")[-1]
    path = f"../data/output/{org_file}_VS_{ref_file}/"
    os.makedirs(path, exist_ok=True)

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
    results = (Parallel(n_jobs=-1)(delayed(cdist_wrapper)(
        metric, path,
        org_latlng_traj_d1, org_latlng_traj_d2, ref_latlng_traj_d1, ref_latlng_traj_d2,
        org_cell_traj_d1, org_cell_traj_d2, ref_cell_traj_d1, ref_cell_traj_d2
        ) for metric in PARAMS["dist"]))
    logger.debug("Parrallell computing done")

    results += (Parallel(n_jobs=-1)(delayed(cell_cdist_wrapper)(
        PARAMS["run"], path,
        org_cell_traj_d1, org_cell_traj_d2, ref_cell_traj_d1, ref_cell_traj_d2,
        alpha
        ) for alpha in PARAMS["alpha"]))

    # list to dictionary
    for i in range(1, len(results)):
        results[0].update(results[i])
    results = results[0]
    logger.debug("transform into dic done")
    logger.info("Computation over")

    # saving results
    df_res = pd.DataFrame(results).T
    df_res.to_csv(f"{path}/results.csv")
    logger.info("Saving over")

if __name__ == "__main__":
    main()
