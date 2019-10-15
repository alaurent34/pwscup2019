import numpy as np
import pandas as pd
from numba import njit

@njit
def get_transition_matrix(array_reg):
    """
    doc
    """
    assert array_reg.ndim == 2

    transition_matrix = np.zeros((1024, 1024))
    for i in range(array_reg.shape[0]):
        for j in range(array_reg.shape[1]-1):
            x = array_reg[i][j] - 1
            y = array_reg[i][j+1] - 1

            transition_matrix[x, y] += 1

    transition_matrix = transition_matrix/(array_reg.shape[0]-1)

    return transition_matrix

@njit
def markov(cell_ano, cell_ref):
    """TODO: Docstring for markov.
    :returns: TODO

    """
    assert cell_ano.ndim == 2
    assert cell_ref.ndim == 2

    transition_matrix = get_transition_matrix(cell_ref)
    cell_ano_markov = cell_ano.copy()

    for i in range(cell_ano.shape[0]):
        for j in range(cell_ano.shape[1]-1):
            x = cell_ano[i][j] - 1
            y = cell_ano[i][j+1] - 1

            if (transition_matrix[x, y] != 0):
                cell_ano_markov[i][j+1] = (y+1)
            else:
                cell_ano_markov[i][j+1] = (x+1)

    return cell_ano_markov
