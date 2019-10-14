"""
File: replace_zeros_prec.py
Author: yourname
Email: yourname@email.com
Github: https://github.com/yourname
Description:
"""
import argparse
import logging
import verboselogs

import pandas as pd
import numpy as np
from numba import njit

def df_traj_to_np(dataframe):
    """TODO: Docstring for df_traj_to_np.
    :returns: TODO

    """
    cell_traj = np.zeros((2000, 400))
    for i in range(dataframe.shape[0]//400):
        cell_traj[i][:] = dataframe.iloc[i*400:(i+1)*400]['reg_id']

    return cell_traj.astype(int)

def np_to_df_traj(array, dataframe):
    """TODO: Docstring for np_to_df_traj.
    :returns: TODO

    """
    assert array.ndim == 2
    assert array.shape[0]*array.shape[1] == dataframe.shape[0]

    dataframe.reg_id = pd.Series(array.flatten())
    return dataframe


@njit
def replace_zero(array_traj):
    """TODO: Docstring for find_next_zero.
    :returns: TODO

    """
    for user in range(array_traj.shape[0]):
        begin = 0
        end = 0
        i = 0
        while i < array_traj[user].shape[0]:
            begin = find_next_zero(i, array_traj[user])
            if begin == array_traj[user].shape[0]:
                break
            end = find_next_end_zero(begin+1, array_traj[user])

            val = find_value_raplace(begin, end, array_traj[user])
            array_traj[user][begin:end] = val

            i = end
    return array_traj

@njit
def find_next_zero(i, array):
    """TODO: Docstring for find_next_zero.
    :returns: TODO

    """
    while array[i] != 0 and i < array.shape[0]:
        i += 1
    return i

@njit
def find_next_end_zero(i, array):
    """TODO: Docstring for find_next_end_zero.
    :returns: TODO

    """
    while array[i] == 0 and i < array.shape[0]:
        i += 1
    return i

@njit
def find_value_raplace(begin, end, array):
    """TODO: Docstring for find_value_raplace.
    :returns: TODO

    """
    if begin == 0 and end == array.shape[0]:
        return np.random.randint(1, 1025)
    if begin == 0 and end != array.shape[0]:
        return array[end]
    return array[begin-1]

# def main():
#     """TODO: Docstring for main.
#     :returns: TODO

#     """
#     # init logger
#     logger = verboselogs.VerboseLogger("attackLogger")
#     logger.addHandler(logging.StreamHandler())
#     logger.setLevel(logging.INFO)

#     # init parser
#     parser = argparse.ArgumentParser("This file preprocess data from PWSCup2019")
#     parser.add_argument("-v", "--verbose", help="Increase output verbosity",
#                         action="store_true")
#     parser.add_argument("-i", "--input", help="Path of the anony traj FILE", type=str,
#                         required=True)

#     # recover args
#     args = parser.parse_args()

#     if args.verbose:
#         print("Verbose mode on")
#         logger.setLevel(logging.DEBUG)

#     logger.debug(f"agrs.type: {args.type}")
#     assert (args.type in ["IDP", "TRP"]), "-t only take IDP or TRP as value"

#     df_traj = pd.read_csv(args.input)

#     cell_traj = df_traj_to_np(df_traj)

# if __name__ == "__main__":
#     main()
