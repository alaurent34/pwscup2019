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

        "alpha": ["0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8",
                  "0.9", "1.0", "2.0", "3.0", "4.0", "round", "home"]
        }

def evaluate(cdist_matrix):
    """TODO: Docstring for evaluate.
    :returns: TODO

    """
    # compare minimum with diagonal of matrix
    min_cdist_m = cdist_matrix.min(axis=1)

    res = (np.count_nonzero(cdist_matrix.diagonal() == min_cdist_m)/
           min_cdist_m.shape[0])

    return res

def cdist_wrapper(metric, path, l_org, l_ref, c_org, c_ref):
    """
    doc
    """
    if metric in PARAMS["lat_lng_dist"]:
        return latlng_cdist_wrapper(metric, path, l_org, l_ref)

    return cell_cdist_wrapper(metric, path, c_org, c_ref)

def cell_cdist_wrapper(metric, path, org, ref, alpha=0):
    """TODO: Docstring for function.

    :arg1: TODO
    :returns: TODO

    """
    # preprocess
    assert ref.shape == org.shape

    upper_bound = max(org.max(), ref.max())

    if metric == "jaccard":
        org = transform_jaccard(org, upper_bound, org.shape[0])
        ref = transform_jaccard(ref, upper_bound, ref.shape[0])
        distance = "jaccard"
    else:
        if alpha != "home":
            org = compute_pfipf(org, upper_bound, org.shape[0])
            ref = compute_pfipf(ref, upper_bound, org.shape[0])
        distance = "cosine"

    # troncature with alpha
    if alpha == "round":
        org = np.round(org)
        ref = np.round(ref)
    elif alpha == "home":
        index_to_replace = []
        for i in range(int(org.shape[1]/20)):
            index_to_replace.append(20*i)
            index_to_replace.append(20*i+1)

        org = org[:, index_to_replace]
        ref = ref[:, index_to_replace]
        print(org)
        org = compute_pfipf(
            org, max(org.max(), ref.max()), org.shape[0])
        ref = compute_pfipf(
            ref, max(org.max(), ref.max()), ref.shape[0])
    else:
        alpha = float(alpha)
        org[np.nonzero(org > alpha)] = 1
        # org_d1d2[np.nonzero(org_d1d2 < alpha)] = 0
        ref[np.nonzero(ref > alpha)] = 1
        # ref_d1d2[np.nonzero(ref_d1d2 < alpha)] = 0

    # computing pairwise distance
    cdist_matrix = cdist(org, ref, metric=distance)


    # saving
    # np.save(f"{path}/{metric}_cdist_matrix_d1.npy", cdist_matrix_d1)
    # np.save(f"{path}/{metric}_cdist_matrix_d2.npy", cdist_matrix_d2)
    np.save(f"{path}/{metric}_cdist_matrix_d1d2_alpha={alpha}.npy", cdist_matrix)

    res = evaluate(cdist_matrix)

    return {f"{metric}_alpha={alpha}": {
                "all": res,
                }
            }

def latlng_cdist_wrapper(metric, path, org, ref):
    """TODO: Docstring for latlng_cdist_wrapper.
    :returns: TODO

    """
    # compute pairwise distance
    if not (metric in ["frechet", "discret_frechet"]):
        cdist_matrix = tdist.cdist(org, ref, metric=metric, type_d="spherical")
    else:
        cdist_matrix = tdist.cdist(org, ref, metric=metric, type_d="euclidean")

    # saving
    np.save(f"{path}/{metric}_cdist_matrix.npy", cdist_matrix)

    res = evaluate(cdist_matrix)

    return {metric: {
                "all": res,
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
    parser.add_argument("-v", "--verbose", help="Increase output verbosity",
                        action="store_true")
    parser.add_argument("-o", "--traj_org", help="Path of the orginal traj directory", type=str,
                        required=True)
    parser.add_argument("-r", "--traj_ref", help="Path of the reference traj directory",
                        type=str, required=True)
    parser.add_argument("-d", "--distance", help="Distance wich will be computed", default=None,
                        type=str)
    parser.add_argument("-p", "--allpfipf", help="Compute all alpha_pfipf",
                        action="store_true")

    # recover args
    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode on")
        logger.setLevel(logging.DEBUG)
    if args.distance:
        PARAMS["dist"] = []
        PARAMS["dist"].append(args.distance)
    logger.debug(f"distances: {PARAMS['dist']}")

    org_file = args.traj_org.split("/")[-1]
    ref_file = args.traj_ref.split("/")[-1]
    path = f"../data/output/{org_file}_VS_{ref_file}/"
    os.makedirs(path, exist_ok=True)

    logger.info("Benchmark is started !")

    # read data
    ## original data set
    # org_df = pd.read_csv(f"{args.traj_org}/dataframe.csv")
    # org_cell_traj_d1 = np.load(f"{args.traj_org}/cell_traj_d1.npy")
    # org_cell_traj_d2 = np.load(f"{args.traj_org}/cell_traj_d2.npy")
    # org_latlng_traj_d1 = np.load(f"{args.traj_org}/lat_lng_traj_d1.npy")
    # org_latlng_traj_d2 = np.load(f"{args.traj_org}/lat_lng_traj_d2.npy")
    org_latlng_traj = np.load(f"{args.traj_org}/lat_lng_traj.npy")
    org_cell_traj = np.load(f"{args.traj_org}/cell_traj.npy")

    ## reference data set
    # ref_df = pd.read_csv(f"{args.traj_ref}/dataframe.csv")
    # ref_cell_traj_d1 = np.load(f"{args.traj_ref}/cell_traj_d1.npy")
    # ref_cell_traj_d2 = np.load(f"{args.traj_ref}/cell_traj_d2.npy")
    # ref_latlng_traj_d1 = np.load(f"{args.traj_ref}/lat_lng_traj_d1.npy")
    # ref_latlng_traj_d2 = np.load(f"{args.traj_ref}/lat_lng_traj_d2.npy")
    ref_latlng_traj = np.load(f"{args.traj_ref}/lat_lng_traj.npy")
    ref_cell_traj = np.load(f"{args.traj_ref}/cell_traj.npy")
    logger.debug("Read numpy arrays (traj)")

    # Computing scores
    results = (Parallel(n_jobs=-1)(delayed(cdist_wrapper)(
        metric, path,
        org_latlng_traj, ref_latlng_traj,
        org_cell_traj, ref_cell_traj
        ) for metric in PARAMS["dist"]))
    logger.debug("First parrallell computing done")

    if args.allpfipf:
        results += (Parallel(n_jobs=-1)(delayed(cell_cdist_wrapper)(
            PARAMS["run"], path,
            org_cell_traj, ref_cell_traj,
            alpha
            ) for alpha in PARAMS["alpha"]))
        logger.debug("Second parrallell computing done")

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
