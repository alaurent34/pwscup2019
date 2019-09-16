"""
File: attack.py
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
from scipy.spatial.distance import cdist

from distance_benchmark import compute_pfipf

def main():
    """ Run distane benchmark
    """
    # init saving dir
    path = f"../data/output/reid_files/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(f"{path}/IDP", exist_ok=True)
    os.makedirs(f"{path}/TRP", exist_ok=True)

    # init logger
    logger = verboselogs.VerboseLogger("attackLogger")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # init parser
    parser = argparse.ArgumentParser("This file preprocess data from PWSCup2019")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity",
                        action="store_true")
    parser.add_argument("-o", "--traj_ano", help="Path of the anony traj DIRECTORY", type=str,
                        required=True)
    parser.add_argument("-r", "--traj_ref", help="Path of the reference traj DIRECTORY",
                        type=str, required=True)
    parser.add_argument("-t", "--type", help="Type of attack either IDP or TRP",
                        type=str, required=True)
    parser.add_argument("-a", "--alpha", help="Alpha value",
                        type=str, default=1)

    # recover args
    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode on")
        logger.setLevel(logging.DEBUG)

    logger.debug(f"agrs.type: {args.type}")
    assert (args.type in ["IDP", "TRP"]), "-t only take IDP or TRP as value"

    if args.type == "IDP":
        # Perform PFIPF attack
        file_name = args.traj_ano.split("/")[-2].split("_")
        file_name[0] = ""
        file_name[1] = "etable"
        file_name = "_".join(file_name[1:])

        logger.info(f"Attack IDP {file_name}")

        # Read files
        cell_ano = np.concatenate([
            np.load(f"{args.traj_ano}/cell_traj_d1.npy"),
            np.load(f"{args.traj_ano}/cell_traj_d2.npy")
            ], axis=1)
        cell_ref = np.concatenate([
            np.load(f"{args.traj_ref}/cell_traj_d1.npy"),
            np.load(f"{args.traj_ref}/cell_traj_d2.npy")
            ], axis=1)

        # Compute pfipf
        pfipf_reg_ano = compute_pfipf(cell_ano, 1024, 2000)
        pfipf_reg_ref = compute_pfipf(cell_ref, 1024, 2000)

        # alpha processing
        pfipf_reg_ano[pfipf_reg_ano >= 1] = 1
        pfipf_reg_ref[pfipf_reg_ref >= 1] = 1

        #distance computing
        versus = cdist(pfipf_reg_ano, pfipf_reg_ref, metric="cosine")

        #Argmin selection
        df = pd.DataFrame(versus.argmin(axis=1)+1, columns=["user_id"])

        #saving
        df.to_csv(f"{path}/IDP/{file_name}.csv", index=False)

    else:
        # Do nothing
        file_name = args.traj_ano.split("/")[-2].split("_")
        file_name[0] = ""
        file_name[1] = "etraces"
        file_name = "_".join(file_name[1:])

        logger.info(f"Attack TRP {file_name}")

        df = pd.read_csv(f"{args.traj_ano}/dataframe.csv")
        df.reg_id.to_csv(f"{path}/TRP/{file_name}.csv", index=False, header="reg_id")


if __name__ == "__main__":
    main()
