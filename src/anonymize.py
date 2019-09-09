"""
File: anonymize.py
Author: Antoine Laurent
Email: laurent.antoine@courrier.uqam.ca
Github: https://github.com/Drayer34
Description: Anonymize files of PWSCUP 2019
"""
import os
import argparse
import logging
import verboselogs

import pandas as pd
import numpy as np

from distance_benchmark import compute_pfipf

PARAMS = {
        "output_path": "../data/output/anon_traces/",
        "anon_file_name": "anotraces_team_011_data",
        "IDP_num": "01",
        "TRP_num": "02"
        }

def pfipf_anonymization(traj, alpha=0.3, random=True):
    """TODO: Docstring for anonymization.
    :returns: TODO

    """
    # compute pfipf matrix
    pfipf_o = compute_pfipf(traj, traj.max(), traj.shape[0])
    # assert that we are not below 0.7 for Utility requirements
    assert (np.count_nonzero(pfipf_o >= alpha)/2000 >= 0.7), "Utility below 0.7"

    # replace all pfipf values above alpha with a random location or -1
    traj_anon = traj.copy()
    for i in range(traj_anon.shape[0]):
        for j in range(traj_anon.shape[1]):
            index = traj_anon[i][j]
            if (pfipf_o[i][index] >= 0.3 or
                    (j < 2 or (traj_anon.shape[1]/2 <= j < (traj_anon.shape[1]/2) + 2))):
                if random:
                    traj_anon[i][j] = np.random.randint(1, 1025)
                else:
                    traj_anon[i][j] = -1

    # transform numpy array into original dataframe
    df_ano = pd.DataFrame(traj_anon)
    df_ano = df_ano.reset_index()
    df_ano["index"] = df_ano.index + 1
    df_ano = df_ano.melt("index").sort_values(["index", "variable"])\
                                 .reset_index(drop=True)\
                                 .drop("variable", axis=1)
    df_ano = df_ano.rename(columns={"index": "user_id", "value": "reg_id"})
    df_ano["reg_id"] = df_ano["reg_id"].astype(str)
    df_ano["reg_id"].replace("-1", "*", inplace=True)

    return df_ano

def main():
    """Run anonymization method
    """
    # init logger
    logger = verboselogs.VerboseLogger("benchLogger")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # init parser
    parser = argparse.ArgumentParser("This file preprocess data from PWSCup2019")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-i", "--input", help="Path of the preprocessed trajectory directory",
                        type=str, required=True)
    parser.add_argument("-o", "--output", help="Path of the anonymized traj directory",
                        type=str, default=PARAMS["output_path"])
    parser.add_argument("-t", "--typed", help="Type of file, either IDP or TPR",
                        type=str, required=True)
    # parser.add_argument("-p", "--pfipf", help="Use pfipf methods", action="store_true")

    # recover args
    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode on")
        logger.setLevel(logging.DEBUG)
    if args.output:
        PARAMS["output_path"] = args.output

    assert (args.type in ["IDP, TPR"]), "-t only take IDP or TPR as value"
    if args.typed == "IDP":
        PARAMS["anon_file_name"] += f"{PARAMS['IDP_num']}_IDP.csv"
    else:
        PARAMS["anon_file_name"] += f"{PARAMS['TRP_num']}_TRP.csv"

    # read the tow days of the trajectory
    o_d1 = np.load(f"{args.input}/cell_traj_d1.npy")
    o_d2 = np.load(f"{args.input}/cell_traj_d2.npy")
    o_d1d2 = np.concatenate([o_d1, o_d2], axis=1)

    # anonymize
    df_ano = pfipf_anonymization(o_d1d2)

    # saving
    os.makedirs(PARAMS["output_path"], exist_ok=True)
    df_ano["reg_id"].to_csv(
        f"{PARAMS['output_path']}/PARAMS['anon_file_name']",
        index=False,
        header="reg_id")

if __name__ == "__main__":
    main()
