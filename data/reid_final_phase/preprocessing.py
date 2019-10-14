"""
File: preprocessing.py
Author: Antoine Laurent
Email: laurent.antoine@courrier.uqam.ca
Github: https://github.com/Drayer34
Description:
"""
import os
import argparse
import logging
import verboselogs

import pandas as pd
import numpy as np

from replace_zeros_prec import df_traj_to_np
from replace_zeros_prec import np_to_df_traj
from replace_zeros_prec import replace_zero

def explode(dataframe, column, sep=' ', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    dataframe : pandas.DataFrame dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `dataframe`.
    """
    indexes = list()
    new_values = list()
    dataframe = dataframe.dropna(subset=[column])
    for i, presplit in enumerate(dataframe[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = dataframe.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df


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
    parser.add_argument("-i", "--input", help="Path of the input file", type=str,
                        required=True)
    parser.add_argument("-r", "--region", default="OrgData(Anony-Pre)_011/info_region.csv",
                        help="Path of file which represent regions", type=str)
    parser.add_argument("-o", "--output", help="Path for the output files (directory)", type=str,
                        required=True)

    # recover args
    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode on")
        logger.setLevel(logging.DEBUG)

    # read files input
    df_traj = pd.read_csv(args.input)
    logger.debug("Read traj")

    # expand generalized trajectories
    expended = False
    if type(df_traj.reg_id.dtype != int):     #####
        expended = True
        df_traj = explode(df_traj, 'reg_id')     #####
        df_traj.reg_id.replace('*', '0', inplace=True)     #####     #####
        df_traj.reg_id = df_traj.reg_id.astype(int)     #####


    # recover trajectories as cells
    if expended:
        cell_traj = df_traj.groupby(["pse_id", "time_id"])["reg_id"]\
                .apply(np.array).unstack().values     #####
    else:
        cell_traj = df_traj_to_np(df_traj)
    logger.debug("Cells trajectories extracted")

    # replace 0s by prev or next value
    cell_traj = replace_zero(cell_traj)
    df_traj = np_to_df_traj(cell_traj, df_traj)

    # save numpy array
    np.save(arr=cell_traj, file=f"{args.output}/cell_traj.npy")
    logger.info("Cells trajectories saved")

    # save dataframe file
    os.makedirs(args.output, exist_ok=True)
    df_traj.to_csv(f"{args.output}/dataframe.csv", index=False)
    logger.info("DataFrame merged saved")

    logger.info("Preprocessing done")

if __name__ == "__main__":
    main()
