import numpy as np
import pandas as pd


def cwm(cell_size, weight, axis=1):
    """
    Community weighted mean of the trait cell size
    :param cell_size: array_like
        phytoplankton size spectra
    :param weight: array_like
        phytoplankton abundance or biomass spectra
    :param axis: integer or tuple of integers, optional
        axis or axes along which the sum is computed.
        See documentation of numpy.nansum
    :return: ndarray
        array with computed community weighted mean
    """
    return np.nansum(cell_size * weight, axis=axis) / np.nansum(weight, axis=axis)


def cwv(cell_size, weight, axis=1):
    """
    Community weighted variance of the trait cell size
    :param cell_size: array_like
        phytoplankton size spectra
    :param weight: array_like
        phytoplankton abundance or biomass spectra
    :param axis: integer or tuple of integers, optional
        axis or axes along which the sum is computed.
        See documentation of numpy.nansum
    :return: ndarray
        array with computed community weighted mean
    """
    weighted_mean = np.nansum(cell_size * weight, axis=axis) / np.nansum(weight, axis=axis)
    return np.nansum(cell_size ** 2 * weight, axis=axis) / np.nansum(weight, axis=axis) - weighted_mean ** 2


def size_var_comp_sbmi(dtf, weight_col='rep_nind',  log_trans=True):
    """
    Size variance components for a community composed of 'i' species
    each with an 'n' number of individuals. Based on the work of
    de Bello et al. (2011 Methods Ecol. & Evol.)
    :param dtf: pandas.DataFrame
                dataframe with output from simulation using any of the size-based models
                'sbm.SBMc', 'sbm.SBMi_syn' or 'sbm.SBMi_asyn'
    :param weight_col: string
                       name of the column with information about abundance or number of individuals
    :param log_trans: bool
                      log-transform cell size

    :return: pandas.DataFrame with mean size and size variance components as columns.
    """
    if log_trans:
        cell_size_i_n = dtf.cell_size.copy()
        log_cell_size_i_n = np.log(dtf.cell_size)
        dtf = dtf.assign(cell_size=log_cell_size_i_n, cell_size_lin=cell_size_i_n)

    no_ind_i = dtf.groupby('spp').sum()[weight_col]
    mean_size_i = dtf.groupby('spp').apply(lambda x: np.nansum(x.cell_size * x[weight_col]) / x[weight_col].sum())
    mean_size_com = np.nansum(no_ind_i / no_ind_i.sum() * mean_size_i)
    between_var = np.nansum(no_ind_i / no_ind_i.sum() * (mean_size_i - mean_size_com) ** 2)
    within_i_ss = dtf.groupby('spp').apply(lambda x: np.nansum(
        x[weight_col] / x[weight_col].sum() * (x.cell_size - (np.nansum(x.cell_size * x[weight_col]) / x[weight_col].sum())) ** 2))
    within_var = np.nansum((no_ind_i / no_ind_i.sum() * within_i_ss))
    tot_var_ss = dtf.groupby('spp').apply(
        lambda x: np.sum(x[weight_col] / x[weight_col].sum() * (x.cell_size - mean_size_com) ** 2))
    tot_var = np.sum((no_ind_i / no_ind_i.sum() * tot_var_ss))

    out_dtf = pd.DataFrame({
        'tot_var': [tot_var],
        'between_var': [between_var],
        'within_var': [within_var],
        'mean_size_com': [mean_size_com]
    })

    spp_cols = {}
    for sp in range(mean_size_i.size):
        spp_cols['name_spp' + str(sp + 1).zfill(2)] = [mean_size_i.index[sp]]
        spp_cols['mean_size_spp' + str(sp + 1).zfill(2)] = [mean_size_i[sp]]

    return out_dtf.join(pd.DataFrame(spp_cols))