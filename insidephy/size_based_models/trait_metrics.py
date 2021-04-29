import numpy as np


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


def size_var_comp_sbmi(cell_size, weight):
    """
    Computation of trait variance components for the trait cell size and
    following de Bello et al. (2011 Methods Ecol. Evol.).
    :param cell_size: array_like
        phytoplankton size spectra as obtained from SBMi model
    :param weight: array_like
        phytoplankton abundance or biomass spectra as obtained from SBMi model
    :return: ndarrays
        arrays with computed total, between and within cell size variance, as well
        as mean cell size for the community and for each species.
    """
    no_ind_i = weight.sum(axis=3)
    tot_no_ind = weight.sum(axis=(1, 3))
    mean_size_i = (cell_size * weight / no_ind_i).sum(axis=3)
    mean_size_com = np.sum(no_ind_i / tot_no_ind * mean_size_i, axis=1)
    tot_var_ss = np.sum((cell_size - mean_size_com) ** 2 * weight / no_ind_i, axis=3)
    tot_var = np.sum(no_ind_i / tot_no_ind * tot_var_ss, axis=1)
    between_var = np.sum(no_ind_i / tot_no_ind * (mean_size_i - mean_size_com) ** 2, axis=1)
    within_var_ss = np.sum((cell_size - mean_size_i) ** 2 * weight / no_ind_i, axis=3)
    within_var = np.sum(no_ind_i / tot_no_ind * within_var_ss, axis=1)
    return tot_var, between_var, within_var, mean_size_com, mean_size_i
