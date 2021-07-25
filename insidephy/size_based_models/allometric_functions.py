import numpy as np


def v_max(cvol):
    """
    Allometric function between cell volume (mum3/cell) and v_max (pgN/cell*day)
    based on Ward et al. (2016 Am. Nat.), with a conversion from pgN/cell*day
    to molN/cell*day.
    :param cvol: cell volume
    :return: maximum uptake rate
    """
    pgNtogN = 1. / 1e12
    gNtomolN = 1. / 14.007
    return (0.024 * cvol ** 1.10) * pgNtogN * gNtomolN  # pgN/cell*d -> molN/cell*d


def q_max(cvol):
    """
    Allometric function between cell volume (mum3/cell) and q_max (pgN/cell)
    based on Marañon et al. (2013 Eco. Lett.), with a conversion from pgN/cell
    to molN/cell
    :param cvol: cell volume
    :return: maximum cell quota
    """
    pgNtogN = 1. / 1e12
    gNtomolN = 1. / 14.007
    return (10 ** (-1.26) * cvol ** 0.93) * pgNtogN * gNtomolN  # pgN/cell -> molN/cell


def q_min(cvol):
    """
    Allometric function between cell volume (mum3/cell) and q_min (pgN/cell)
    based on Ward et al. (2016 Am. Nat.), with a conversion from pgN/cell
    to molN/cell
    :param cvol: cell volume
    :return: minimum cell quota
    """
    pgNtogN = 1. / 1e12
    gNtomolN = 1. / 14.007
    return (0.032 * cvol ** 0.76) * pgNtogN * gNtomolN  # pgN/cell -> molN/cell


def mu_inf(cvol):
    """
    Allometric function between cell volume (mum3/cell) and mu_inf (1/day)
    based on Ward et al. (2016 Am. Nat.).
    :param cvol: cell volume
    :return: growth rate at infinite cell quota
    """
    return 4.7 * cvol ** -0.26  # 1/d


def mu_max(cvol):
    """
    Function between cell volume (mum3/cell) and mu_max (1/day)
    based on Ward et al. (2016 Am. Nat.).
    :param cvol: cell volume
    :return: maximum growth rate
    """
    muinf = mu_inf(cvol)
    vmax = v_max(cvol)
    qmin = q_min(cvol)
    return (muinf * vmax) / ((muinf * qmin) + vmax)  # 1/d


def k_r(cvol):
    """
    Allometric function between cell volume (mum3/cell) and K_N (mumolN/L)
    based on Edwards et al. (2012) L&O, with a conversion from mumolN/L to
    molN/L.
    :param cvol: cell volume
    :return: Nitrogen half-saturation constant
    """
    umoltomol = 1. / 1e6
    return (10 ** (-0.84) * cvol ** 0.33) * umoltomol  # mumol N/L -> molN/L


def biomass(cvol):
    """
    Allometric function between cell volume (mum3/cell) and biomass (pgC/cell)
    based on Marañon et al. (2013 Eco. Lett.) with a conversion from pgC/cell
    to molC/cell.
    :param cvol: cell volume
    :return: Cell biomass
    """
    pgCtogC = 1. / 1e12
    gCtomolC = 1. / 12.0107
    return (10 ** (-0.69) * cvol ** 0.88) * pgCtogC * gCtomolC  # pgC/cell -> molC/cell


def biomass_to_size(biomass):
    """
    Inverse conversion from biomass to cell size using the allometric function
    between cell volume (um3/cell) and biomass (C biomass pgC/cell) based on
    Marañón et al. (2013 Eco. Lett.) with a conversion from molC/cell to pgC/cell.
    """
    molCtogC = 12.0107 / 1.
    gCtopgC = 1e12 / 1.
    pgCbiomass = biomass * molCtogC * gCtopgC  # molC/cell -> pgC/cell
    return 10 ** ((np.log10(pgCbiomass) - -0.69) / 0.88)  # (um^3/cell)
