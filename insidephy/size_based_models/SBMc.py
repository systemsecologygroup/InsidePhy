import numpy as np
import time
from scipy.integrate import odeint
import insidephy.size_based_models.allometric_functions as allo


class SBMc:
    """
    Size-based model of phytoplankton size classes.
    :param ini_resource: float
        initial value resource concentration in units of nitrogen (M N)
    :param ini_density: list or tuple of floats
        initial density of cells in number of cells per litre
    :param spp_names: list or tuple of floats
        two letter name tag per species or strain
    :param min_size: list or tuple of floats
        minimum cell size per species or strain
    :param max_size: list or tuple of floats
        maximum cell size per species or strain
    :param num_sc: list or tuple of integers
        number of size classes per species or strain used in SBMc
    :param dilution_rate: float
        rate of medium exchange in the culture system
    :param volume: float
        volume of the culture system
    :param time_ini: integer
        initial time used in the simulation
    :param time_end: integer
        final time in days used in the simulation
    :param time_steps: integer
        time steps used in the simulations
    :param timeit: bool
        time of the simulation
    :param vectorize: bool
        whether to calculate system of equations with a loop
        or with numpy array operations
    :return: object with solution of size-based model
    """

    def __init__(self, ini_resource, ini_density, spp_names, min_size, max_size, num_sc,
                 dilution_rate, volume, time_ini=0, time_end=20, time_steps=100, timeit=False, vectorize=True):
        if not all([isinstance(item, list) or isinstance(item, tuple)
                    for item in iter([spp_names, ini_density, min_size, max_size, num_sc])]):
            raise TypeError('Error on input parameters spp_names, ini_density, min_size, max_size or num_sc. '
                            'Input parameters must be type list or tuple.')
        if not all([len(lst) == len(spp_names) for lst in iter([spp_names, ini_density, min_size, max_size, num_sc])]):
            raise ValueError("initial values of spp_names, ini_density, min_size, max_size and num_sc "
                             "must be lists of the same length depending on the number of species use "
                             "in the simulation")

        # Simulation parameters and variables
        start_comp_time = time.time()
        self._vectorize = vectorize
        self._ini_resource = ini_resource
        self._ini_density = np.concatenate(([[ini_density[i] / n for l in range(n)] for i, n in enumerate(num_sc)]), 0)
        self._minsize = min_size
        self._maxsize = max_size
        self._numsc = num_sc
        self._spp_names = spp_names
        self._dilution_rate = dilution_rate
        self._volume = volume
        self._size_range = np.concatenate(([np.logspace(np.log10(min_size[n]), np.log10(max_size[n]), num_sc[n])
                                            for n, l in enumerate(spp_names)]), 0)
        self._ini_quota = (allo.q_max(self._size_range) +
                           allo.q_min(self._size_range)) / 2.
        self._y0 = np.concatenate(([self._ini_resource],
                                   self._ini_quota,
                                   self._ini_density,
                                   np.zeros(sum(num_sc))), 0)
        # output arrays
        self.time = np.linspace(time_ini, time_end, time_steps)
        solution = odeint(self._syseqns, self._y0, self.time, rtol=1e-12, atol=1e-12)
        self.resource = solution[:, 0]
        self.quota = solution[:, 1:sum(num_sc) + 1]
        self.abundance = solution[:, 1 + sum(num_sc): 1 + sum(num_sc) * 2]
        self.biomass = self.abundance * allo.biomass(self._size_range)
        self.mus = solution[1:, 1 + sum(num_sc) * 2:] - solution[0:-1, 1 + sum(num_sc) * 2:]
        self.mus = np.append(self.mus, self.mus[-1, :].reshape(1, sum(num_sc)), 0)

        if timeit:
            print('SBMc total simulation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))

    def _syseqns(self, y, t):
        """
        System of ordinary differential equations
        :param y: array with initial conditions
        :param t: time array
        :return: solution of ode system
        """
        R = y[0]  # resource in units of Nitrogen (uM N)
        Q = y[1:sum(self._numsc) + 1]  # Quota (molN/molC*cell)
        N = y[sum(self._numsc) + 1: 2 * sum(self._numsc) + 1]  # Cell density (cells/L)
        S = self._size_range  # Cell size (mum^3)
        drqndt = np.zeros(len(y))  # Array with solution of the system of equations

        if self._vectorize:
            vmaxsc = allo.v_max(S)  # Maximum uptake rate
            qminsc = allo.q_min(S)  # Minimum internal quota
            qmaxsc = allo.q_max(S)  # Maximum internal quota
            mumaxsc = allo.mu_max(S)  # Maximum growth rate
            knsc = allo.k_r(S)  # Resource half-saturation
            vsc = vmaxsc * ((qmaxsc - Q) / (qmaxsc - qminsc)) * (R / (R + knsc))  # uptake
            muc = mumaxsc * (1. - (qminsc / Q))  # growth
            drqndt[0] = self._dilution_rate * (self._ini_resource - R) - np.sum(vsc * N)  # Resources
            drqndt[1:sum(self._numsc) + 1] = vsc - muc * Q  # Internal Quota
            drqndt[sum(self._numsc) + 1: 2 * sum(self._numsc) + 1] = (muc - self._dilution_rate) * N  # Cell Density
            drqndt[2 * sum(self._numsc) + 1:] = muc
        else:
            drqndt[0] = self._dilution_rate * (self._ini_resource - R)  # Resources
            for sc in range(sum(self._numsc)):
                Qsc = Q[sc]  # Internal quota of size class j
                Nsc = N[sc]  # Cell density of size class j
                Ssc = S[sc]  # Cell volume of size class j
                vmaxsc = allo.v_max(Ssc)  # Maximum uptake rate
                qminsc = allo.q_min(Ssc)  # Minimum internal quota
                qmaxsc = allo.q_max(Ssc)  # Maximum internal quota
                mumaxsc = allo.mu_max(Ssc)  # Maximum growth rate
                knsc = allo.k_r(Ssc)  # Resource half-saturation
                vsc = vmaxsc * ((qmaxsc - Qsc) / (qmaxsc - qminsc)) * (R / (R + knsc))  # uptake
                muc = mumaxsc * (1. - (qminsc / Qsc))  # growth
                drqndt[0] -= vsc * Nsc  # Resources
                drqndt[1 + sc] = vsc - muc * Qsc  # Internal Quota
                drqndt[1 + sum(self._numsc) + sc] = (muc - self._dilution_rate) * Nsc  # Cell Density
                drqndt[1 + sum(self._numsc) * 2 + sc] = muc
        return drqndt
