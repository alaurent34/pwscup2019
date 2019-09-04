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
    df_region = pd.read_csv(args.region)
    logger.debug("Read traj and region files")

    # merge region with trajectories
    df_traj = df_traj.merge(df_region, on="reg_id")
    df_traj.sort_values(by=["user_id", "time_id"], inplace=True)
    df_traj.reset_index(drop=True, inplace=True)
    logger.debug("Merge accomplished")

    # save dataframe file
    os.makedirs(args.output, exist_ok=True)
    df_traj.to_csv(f"{args.output}/dataframe.csv", index=False)
    logger.info("DataFrame merged saved")

    # recover trajectories as cells
    cell_traj = np.array(list(df_traj.groupby("user_id")["reg_id"].apply(np.array)))
    one_day_time = int(cell_traj.shape[1]/2)
    cell_traj_first_d = cell_traj[:, 0:one_day_time]
    cell_traj_scnd_d = cell_traj[:, one_day_time:]
    logger.debug("Cells trajectories extracted")

    np.save(arr=cell_traj_first_d, file=f"{args.output}/cell_traj_d1.npy")
    np.save(arr=cell_traj_scnd_d, file=f"{args.output}/cell_traj_d2.npy")
    logger.info("Cells trajectories saved")

    #recover trajectories as lat, lng
    lat_lng_traj = np.array(
        list(df_traj.groupby("user_id").apply(lambda x: np.array(x[["y(center)", "x(center)"]])))
        )
    lat_lng_traj_first_d = lat_lng_traj[:, 0:one_day_time]
    lat_lng_traj_scnd_d = lat_lng_traj[:, one_day_time:]
    logger.debug("Lat, Lng trajectories extracted")

    np.save(arr=lat_lng_traj_first_d, file=f"{args.output}/lat_lng_traj_d1.npy")
    np.save(arr=lat_lng_traj_scnd_d, file=f"{args.output}/lat_lng_traj_d2.npy")
    logger.info("Lat, Lng trajectories saved")

    logger.info("Preprocessing done")

if __name__ == "__main__":
    main()
