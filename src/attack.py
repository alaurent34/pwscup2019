"""
File: attack.py
Author: Antoine Laurent
Email: laurent.antoine@courrier.uqam.ca
Github: https://github.com/Drayer34
Description: Anonymize files of PWSCUP 2019
"""
import os
import math
import argparse
import logging
import verboselogs

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from numba import njit

from distance_benchmark import compute_pfipf
from distance_benchmark import transform_jaccard
from markov import markov

@njit
def create_mapping_trp(versus):
    """TODO: Docstring for function.

    :arg1: TODO
    :returns: TODO

    """
    # reconstrcut mapping
    mapping = np.zeros(2000)
    for i in range(2000):
        pse = versus[i].T.argmin()
        mapping[i] = versus[i].T.argmin() + 2001
        versus[pse] = 2

    return mapping

def mv_cell(cell_number, x, y):
    """
    doc
    """
    x_pos = cell_number % 32
    x_pos = 32 if x_pos == 0 else x_pos
    y_pos = int((cell_number - 1) / 32)

    #assert 1 <= x_pos + x < 33, "depassing x limits"
    #assert 0 <= y_pos + y < 32, "depassing y limits"
    if not 1 <= x_pos + x < 33 or not 0 <= y_pos + y < 32:
        return cell_number

    x_pos += x
    y_pos += y

    return x_pos + 32*(y_pos)

def create_mapping(size):
    """
    doc
    """
    map_cell = {}
    assert 32 % size == 0, "cannot shrink map to size"
    for i in range(32//size):
        for j in range(32//size):
            cell_left_above = (j*size)+1 + (i*math.pow(size, 2)*(32//size))
            ind_cell_map = 32//size*i+j
            map_cell[ind_cell_map] = []
            #do something with i
            for k in range(size):
                for l in range(size):
                    map_cell[ind_cell_map].append(mv_cell(cell_left_above, k, l))

    mapping = pd.DataFrame(map_cell).T.reset_index().melt("index")\
                .drop("variable", axis=1).sort_values("value").reset_index(drop=True)
    mapping.columns = ["reg_id_ext", "reg_id"]

    return np.array(mapping.reg_id_ext).astype(int)+1

@njit
def mapp_cell_to_extended(array, reg_id_ext):
    """
    doc
    """
    array = array.copy()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i][j] = reg_id_ext[array[i][j]-1]

    return array

def main():
    """ Run distane benchmark
    """
    # init saving dir
    path = f"../data/output/reid_files/"
    # path = "data_iddisclo"
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
    # parser.add_argument("-t", "--type", help="Type of attack either IDP or TRP",
    #                     type=str, required=True)
    parser.add_argument("-a", "--alpha", help="Alpha value",
                        type=int, default=10)
    parser.add_argument("-j", "--jaccard", help="Use Jaccard instead of pf-ipf",
                        action="store_true")
    parser.add_argument("-s", "--size", help="select size extension", type=int, default=1)

    # recover args
    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode on")
        logger.setLevel(logging.DEBUG)

    # logger.debug(f"agrs.type: {args.type}")
    # assert (args.type in ["IDP", "TRP"]), "-t only take IDP or TRP as value"

    # Read files
    cell_ano = np.load(f"{args.traj_ano}/cell_traj.npy")
    cell_ref = np.load(f"{args.traj_ref}/cell_traj.npy")

    cell_ano = markov(cell_ano, cell_ref)

    ######################
    # Perform IDP attack #
    ######################

    file_name = args.traj_ano.split("/")[-2].split("_")
    file_name[0] = ""
    file_name[1] = "etable"
    number = file_name[2][4:]
    file_name[2] = f"team011-{number}"
    file_name = "_".join(file_name[1:])
    # file_name = "etable_team011_data02_IDP"

    logger.info(f"Attack IDP {file_name}")

    reg_id_ext = create_mapping(args.size)
    cell_ano_ext = mapp_cell_to_extended(cell_ano, reg_id_ext)
    cell_ref_ext = mapp_cell_to_extended(cell_ref, reg_id_ext)

    if not args.jaccard:
        # Compute pfipf
        pfipf_reg_ano = compute_pfipf(cell_ano_ext, 1024, 2000)
        pfipf_reg_ref = compute_pfipf(cell_ref_ext, 1024, 2000)

        # alpha processing
        pfipf_reg_ano[pfipf_reg_ano >= args.alpha] = 1
        pfipf_reg_ref[pfipf_reg_ref >= args.alpha] = 1

        #distance computing
        versus = cdist(pfipf_reg_ano, pfipf_reg_ref, metric="cosine")

    else:
        # Compute Jaccard
        jaccard_reg_ano = transform_jaccard(cell_ano_ext, 1024, 2000)
        jaccard_reg_ref = transform_jaccard(cell_ref_ext, 1024, 2000)

        versus = cdist(jaccard_reg_ano, jaccard_reg_ref, metric="jaccard")

    #Argmin selection
    df = pd.DataFrame(versus.argmin(axis=1)+1, columns=["user_id"])

    #saving
    df.to_csv(f"{path}/IDP/{file_name}.csv", index=False)

    ######################
    # Perform TRP attack #
    ######################

    file_name = args.traj_ano.split("/")[-2].split("_")
    file_name[0] = ""
    file_name[1] = "etraces"
    number = file_name[2][4:]
    file_name[2] = f"team011-{number}"
    file_name = "_".join(file_name[1:])

    logger.info(f"Attack TRP {file_name}")

    # reconstruct anon trace dataframe
    df_ano = pd.DataFrame(cell_ano)
    df_ano = df_ano.reset_index()
    df_ano["index"] += 2001
    df_ano = df_ano.melt("index").sort_values(["index", "variable"])\
            .reset_index(drop=True).drop("variable", axis=1)
    df_ano = df_ano.rename(columns={"index": "pse_id", "value": "reg_id"})

    mapping = create_mapping_trp(versus)
    df = pd.DataFrame({"pse_id": mapping})
    df = df.reset_index()
    df["index"] += 1
    df = df.rename(columns={"index": "user_id"})

    # merge
    df_ano = pd.merge(df_ano, df, on="pse_id")
    df_ano = df_ano.reset_index().sort_values(["user_id", "pse_id", "index"])

    # saving
    df_ano.reg_id.to_csv(f"{path}/TRP/{file_name}.csv", index=False, header=["reg_id"])


if __name__ == "__main__":
    main()
