import numpy as np
import time
from scipy.integrate import odeint
import insidephy.size_based_models.allometric_functions as AlloFunc


class SBMc:
    """
    Size-based model of a group of phytoplankton cells within a size class.
    :return: object with solution of size-based model
    """

    def __init__(self, ini_resource, ini_density, minsize, maxsize, spp_names, numsc,
                 dilution_rate, volume, t0=0, tend=20, tsteps=100, timeit=False, vectorize=True):

        start_comp_time = time.time()
        self.vectorize = vectorize
        self.ini_resource = ini_resource
        if not all(len(lst) == len(spp_names) for lst in iter([spp_names, ini_density, minsize, maxsize, numsc])):
            raise ValueError("initial values for species must be lists of the same length")
        self.ini_density = np.concatenate(([[ini_density[i]/n for l in range(n)] for i, n in enumerate(numsc)]), 0)
        self.minsize = minsize
        self.maxsize = maxsize
        self.numsc = numsc
        self.spp_names = spp_names
        self.dilution_rate = dilution_rate
        self.volume = volume
        self.size_range = np.concatenate(([np.logspace(np.log10(minsize[n]), np.log10(maxsize[n]), numsc[n])
                                         for n, l in enumerate(spp_names)]), 0)
        self.ini_quota = (AlloFunc.q_max(self.size_range) +
                          AlloFunc.q_min(self.size_range)) / 2.
        self.y0 = np.concatenate(([self.ini_resource],
                                  self.ini_quota,
                                  self.ini_density,
                                  np.zeros(sum(numsc))), 0)
        self.time = np.linspace(t0, tend, tsteps)
        solution = odeint(self.syseqns, self.y0, self.time, rtol=1e-12, atol=1e-12)
        self.resource = solution[:, 0]
        self.quota = solution[:, 1:sum(numsc)+1]
        self.abundance = solution[:, 1+sum(numsc): 1+sum(numsc)*2]
        self.biomass = self.abundance * AlloFunc.biomass(self.size_range)
        self.mus = solution[1:, 1+sum(numsc)*2:] - solution[0:-1, 1+sum(numsc)*2:]
        self.mus = np.append(self.mus, self.mus[-1,:].reshape(1, sum(numsc)), 0) #, np.tile(np.nan, sum(numsc)).reshape(1, sum(numsc)), 0)

        if timeit:
            print('SBMc total simulation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))

    def syseqns(self, y, t):
        """
        System of ordinary differential equations
        :param y: array with initial conditions
        :param t: time array
        :return: solution of ode system
        """
        R = y[0]  # resource in units of Nitrogen (uM N)
        Q = y[1:sum(self.numsc) + 1]  # Quota (molN/molC*cell)
        N = y[sum(self.numsc) + 1: 2 * sum(self.numsc) + 1]  # Cell density (cells/L)
        S = self.size_range  # Cell size (mum^3)
        drqndt = np.zeros(len(y))  # Array with solution of the system of equations

        if self.vectorize:
            vmaxsc = AlloFunc.v_max(S)  # Maximum uptake rate
            qminsc = AlloFunc.q_min(S)  # Minimum internal quota
            qmaxsc = AlloFunc.q_max(S)  # Maximum internal quota
            mumaxsc = AlloFunc.mu_max(S)  # Maximum growth rate
            knsc = AlloFunc.k_r(S)  # Resource half-saturation
            vsc = vmaxsc * ((qmaxsc - Q) / (qmaxsc - qminsc)) * (R / (R + knsc))  # uptake
            muc = mumaxsc * (1. - (qminsc / Q))  # growth
            drqndt[0] = self.dilution_rate * (self.ini_resource - R) - np.sum(vsc * N)  # Resources
            drqndt[1:sum(self.numsc) + 1] = vsc - muc * Q  # Internal Quota
            drqndt[sum(self.numsc) + 1: 2 * sum(self.numsc) + 1] = (muc - self.dilution_rate) * N  # Cell Density
            drqndt[2 * sum(self.numsc) + 1:] = muc
        else:
            drqndt[0] = self.dilution_rate * (self.ini_resource - R)  # Resources
            for sc in range(sum(self.numsc)):
                Qsc = Q[sc]  # Internal quota of size class j
                Nsc = N[sc]  # Cell density of size class j
                Ssc = S[sc]  # Cell volume of size class j
                vmaxsc = AlloFunc.v_max(Ssc)  # Maximum uptake rate
                qminsc = AlloFunc.q_min(Ssc)  # Minimum internal quota
                qmaxsc = AlloFunc.q_max(Ssc)  # Maximum internal quota
                mumaxsc = AlloFunc.mu_max(Ssc)  # Maximum growth rate
                knsc = AlloFunc.k_r(Ssc)  # Resource half-saturation
                vsc = vmaxsc * ((qmaxsc - Qsc) / (qmaxsc - qminsc)) * (R / (R + knsc))  # uptake
                muc = mumaxsc * (1. - (qminsc / Qsc))  # growth
                drqndt[0] -= vsc * Nsc  # Resources
                drqndt[1 + sc] = vsc - muc * Qsc  # Internal Quota
                drqndt[1 + sum(self.numsc) + sc] = (muc - self.dilution_rate) * Nsc  # Cell Density
                drqndt[1 + sum(self.numsc)*2 + sc] = muc
        return drqndt
