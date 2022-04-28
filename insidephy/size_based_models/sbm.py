import numpy as np
import pandas as pd
from scipy.integrate import odeint
import insidephy.size_based_models.allometric_functions as allo
import textwrap


class SBMbase:
    def __init__(self, ini_resource, ini_density, spp_names, min_cell_size, max_cell_size, dilution_rate, volume):
        """
        Base class for size-based models of phytoplankton cells.
        :param ini_resource: float
        initial value resource concentration in units of nitrogen (M N)
        :param ini_density: list or tuple of floats
        initial density of cells in number of cells per litre
        :param spp_names: list or tuple of floats
        two letter name tag per species or strain
        :param min_cell_size: list or tuple of floats
        minimum cell size per species or strain
        :param max_cell_size: list or tuple of floats
        maximum cell size per species or strain
        :param dilution_rate: float
        rate of medium exchange in the culture system
        :param volume: float
        volume of the culture system
        """
        if not all([isinstance(item, list) or isinstance(item, tuple)
                    for item in iter([spp_names, ini_density, min_cell_size, max_cell_size])]):
            raise TypeError('Error on input parameters spp_names, ini_density, min_cell_size, or max_cell_size. '
                            'Input parameters must be type list or tuple.')
        if not all([len(lst) == len(spp_names) for lst in
                    iter([spp_names, ini_density, min_cell_size, max_cell_size])]):
            raise ValueError("initial values of spp_names, ini_density, min_cell_size, or max_cell_size"
                             "must be lists of the same length depending on the number of species use "
                             "in the simulation")
        self._params = {
            'ini_resource': ini_resource,
            'ini_density': ini_density,
            'spp_names': spp_names,
            'min_cell_size': min_cell_size,
            'max_cell_size': max_cell_size,
            'dilution_rate': dilution_rate,
            'volume': volume
        }

    def __repr__(self):
        class_str = type(self).__name__
        params_str = "\n".join(["{}: {}".format(k, v)
                                for k, v in self._params.items()])
        return "<({})>\nParameters:\n{}\n".format(
            class_str, textwrap.indent(params_str, '    '))


class SBMc(SBMbase):
    def __init__(self, ini_resource, ini_density, spp_names, min_cell_size, max_cell_size, dilution_rate, volume,
                 num_sc, time_ini=0, time_end=20, time_step=100, vectorize=True):
        """
            Size-based model of phytoplankton size classes.
            :param num_sc: list or tuple of integers
                number of size classes per species or strain used in SBMc
            :param time_ini: integer
                initial time used in the simulation
            :param time_end: integer
                final time in days used in the simulation
            :param time_step: integer
                time steps used in the simulations
            :param vectorize: bool
                whether to calculate system of equations with a loop
                or with numpy array operations
            :return: size-based model with solution
            """
        super().__init__(ini_resource, ini_density, spp_names, min_cell_size, max_cell_size, dilution_rate, volume)
        self._vectorize = vectorize
        # Parameters
        self._params = {
            'ini_resource': ini_resource,
            'ini_density': ini_density,
            'spp_names': spp_names,
            'min_cell_size': min_cell_size,
            'max_cell_size': max_cell_size,
            'num_sc': num_sc,
            'dilution_rate': dilution_rate,
            'volume': volume
        }
        # initial conditions
        _size_range = np.concatenate(([np.logspace(np.log10(min_cell_size[n]), np.log10(max_cell_size[n]), num_sc[n])
                                       for n in range(len(spp_names))]), 0)
        self._ini_cond = {
            'ini_resource': ini_resource,
            'ini_density': np.concatenate(([[ini_density[i] / n for l in range(n)] for i, n in enumerate(num_sc)]), 0),
            'size_range': _size_range,
            'ini_quota': (allo.q_max(_size_range) + allo.q_min(_size_range)) / 2.
        }
        self._time_ini = time_ini
        self._time_end = time_end
        self._time_step = time_step

        # run simulation
        self._run()
        self.dtf = self._to_DataFrame()

    def _run(self):
        """
        Initialize and run
        :return: solution of numerical simulation
        """
        _y0 = np.concatenate(([self._ini_cond['ini_resource']],
                              self._ini_cond['ini_quota'],
                              self._ini_cond['ini_density'],
                              np.zeros(np.sum(self._params['num_sc']))), 0)
        _time = np.linspace(self._time_ini, self._time_end, self._time_step)
        self._solution = odeint(self._sys_eqns, _y0, _time, rtol=1e-12, atol=1e-12)

    def _to_DataFrame(self):
        """
        Transform arroy with solution into pandas dataframe
        :return: pandas.DataFrame
        """
        sum_num_sc = np.sum(self._params['num_sc'])
        resource = np.repeat(self._solution[:, 0], sum_num_sc)
        cell_size = np.tile(self._ini_cond['size_range'], self._time_step)
        quota = self._solution[:, 1:sum_num_sc + 1].flatten()
        abundance = self._solution[:, 1 + sum_num_sc:1 + sum_num_sc * 2].flatten()
        time = np.linspace(self._time_ini, self._time_end, self._time_step)
        dtf = pd.DataFrame(
            {'time': np.repeat(time, sum_num_sc),
             'resource': resource,
             'spp': np.tile(np.repeat(self._params['spp_names'], self._params['num_sc']), self._time_step),
             'cell_size': cell_size,
             'quota': quota,
             'abundance': abundance,
             'biomass': abundance * allo.biomass(cell_size),
             }
        )
        return dtf

    def _sys_eqns(self, y, t):
        """
        System of ordinary differential equations
        :param y: array with initial conditions
        :param t: time array
        :return: solution of ode system
        """
        sum_num_sc = np.sum(self._params['num_sc'])
        R = y[0]  # resource in units of Nitrogen (uM N)
        Q = y[1:sum_num_sc + 1]  # Quota (molN/molC*cell)
        N = y[sum_num_sc + 1: 2 * sum_num_sc + 1]  # Cell density (cells/L)
        S = self._ini_cond['size_range']  # Cell size (mum^3)
        drqndt = np.zeros(len(y))  # Array with solution of the system of equations

        if self._vectorize:
            vmaxsc = allo.v_max(S)  # Maximum uptake rate
            qminsc = allo.q_min(S)  # Minimum internal quota
            qmaxsc = allo.q_max(S)  # Maximum internal quota
            mumaxsc = allo.mu_max(S)  # Maximum growth rate
            knsc = allo.k_r(S)  # Resource half-saturation
            vsc = vmaxsc * ((qmaxsc - Q) / (qmaxsc - qminsc)) * (R / (R + knsc))  # uptake
            muc = mumaxsc * (1. - (qminsc / Q))  # growth
            drqndt[0] = self._params['dilution_rate'] * (self._ini_cond['ini_resource'] - R) - np.sum(
                vsc * N)  # Resources
            drqndt[1:sum_num_sc + 1] = vsc - muc * Q  # Internal Quota
            drqndt[sum_num_sc + 1: 2 * sum_num_sc + 1] = (muc - self._params['dilution_rate']) * N  # Cell Density
            drqndt[2 * sum_num_sc + 1:] = muc
        else:
            drqndt[0] = self._params['dilution_rate'] * (self._ini_cond['ini_resource'] - R)  # Resources
            for sc in range(sum_num_sc):
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
                drqndt[1 + sum_num_sc + sc] = (muc - self._params['dilution_rate']) * Nsc  # Cell Density
                drqndt[1 + sum_num_sc * 2 + sc] = muc
        return drqndt


class SBMi_syn(SBMbase):
    def __init__(self, ini_resource, ini_density, spp_names, min_cell_size, max_cell_size, nsi_spp, nsi_min, nsi_max,
                 dilution_rate, volume, time_end=20, time_step=1 / 24, print_time_step=0.5, random_seed=1234):
        """
        Size-based model of individual phytoplankton cells with synchronous updating of agents.
        :param nsi_spp: list or tuple of integers
        number of super individuals per species or strain
        :param nsi_min: integer
        minimum number of super individual for all species and strain
        :param nsi_max: integer
        maximum number of super individual for all species and strain
        :param time_end: integer
        final time in days used in the simulations
        :param time_step: float
        time steps used in the simulations
        :param print_time_step: integer
        time step in days used to store results

        :return: size-based model with solution
        """
        super().__init__(ini_resource, ini_density, spp_names, min_cell_size, max_cell_size, dilution_rate, volume)
        self.spp_si = {}
        self._rng = np.random.default_rng(random_seed)
        self._params = {
            'ini_resource': ini_resource,
            'spp_names': spp_names,
            'ini_nind': ini_density,
            'min_cell_size': min_cell_size,
            'max_cell_size': max_cell_size,
            'nsi_spp': list(nsi_spp),
            'nsi_min': nsi_min,
            'nsi_max': nsi_max,
            'dilution_rate': dilution_rate,
            'volume': volume
        }
        self.dtf = pd.DataFrame({
            'time': None,
            'resource': None,
            'spp': np.array([]),
            'id': np.array([]),
            'cell_size': np.array([]),
            'q_max': np.array([]),
            'q_min': np.array([]),
            'v_max': np.array([]),
            'mu_max': np.array([]),
            'kr': np.array([]),
            'ini_biomass': np.array([]),
            'biomass': np.array([]),
            'quota': np.array([]),
            'rep_nind': np.array([])
        })
        self._time_step = time_step
        self._dtp = print_time_step
        self._time_end = time_end
        self.run()

    def initialize(self):

        spp_size_spectra = np.concatenate([10 ** self._rng.uniform(np.log10(minsize),
                                                                   np.log10(maxsize),
                                                                   size=ag)
                                           for minsize, maxsize, ag in
                                           zip(self._params['min_cell_size'],
                                               self._params['max_cell_size'],
                                               self._params['nsi_spp'])])

        ids = np.concatenate([np.arange(ag) for ag in self._params['nsi_spp']])
        ini_biomass = allo.biomass(spp_size_spectra)
        q_max = allo.q_max(spp_size_spectra) / ini_biomass
        q_min = allo.q_min(spp_size_spectra) / ini_biomass
        spp_super_individuals = {
            'time': 0,
            'resource': self._params['ini_resource'],
            'spp': np.repeat(self._params['spp_names'], self._params['nsi_spp']),
            'id': ids,
            'cell_size': spp_size_spectra,
            'q_max': q_max,
            'q_min': q_min,
            'v_max': allo.v_max(spp_size_spectra) / ini_biomass,
            'mu_max': allo.mu_max(spp_size_spectra),
            'kr': allo.k_r(spp_size_spectra),
            'ini_biomass': ini_biomass,
            'biomass': ini_biomass,
            'quota': (q_max + q_min) / 2,
            'rep_nind': np.repeat(np.array(self._params['ini_nind']) // np.array(self._params['nsi_spp']),
                                  self._params['nsi_spp'])
        }
        self.spp_si.update(spp_super_individuals)

    def update(self):
        next_time_step = self.spp_si['time'] + self._time_step
        # print('time: ', next_time_step)
        # Individuals growth
        self._growth()
        # Reproduction
        self._births()
        # Dilution lost
        self._deaths()
        self.spp_si['time'] = next_time_step
        # print(self.spp_si)
        self._split_combine()

    def run(self):
        simtime = self._time_step  # Simulation time
        printtime = self._dtp  # Print time step
        indexpt = 1  # index of print time step to save results
        self.initialize()
        self.dtf = self.dtf.append(self.to_DataFrame())
        while simtime <= self._time_end + self._time_step:
            self.update()  # Update state of all agents
            if simtime >= printtime:  # Save results
                # print(self.spp_si)
                self.dtf = self.dtf.append(self.to_DataFrame())
                printtime += self._dtp
                indexpt += 1
            simtime += self._time_step  # increase simulation time by one time step (fraction of a day)

    def _growth(self):
        R = np.max([self.spp_si['resource'], 0])
        # R = self.spp_si['resource']
        resource_gains = (self._params['dilution_rate'] * (self._params['ini_resource'] - R)) * \
                         self._time_step
        spp_si = {}
        spp_si_V = self.spp_si['v_max'] * ((self.spp_si['q_max'] - self.spp_si['quota']) /
                                           (self.spp_si['q_max'] - self.spp_si['q_min'])) * \
                   (R / (self.spp_si['kr'] + R))
        #           = molN/molC*day * ([molN/molC - molN/molC]/[molN/molC - molN/molC]) * (MN/[MN + MN])
        #           = molN/molC*day
        resource_loss = spp_si_V * self.spp_si['biomass'] * self.spp_si['rep_nind'] * self._time_step / self._params[
            'volume']
        #                     = molN/molC*day * molC/cell * cell * day/L
        #                     = molN/L
        spp_si_mu = self.spp_si['mu_max'] * (1. - (self.spp_si['q_min'] / self.spp_si['quota']))  # 0.
        #            = 1/day * (1. - molN/molC / molN/molC)
        #            = 1/day
        spp_si['quota'] = self.spp_si['quota'] + (spp_si_V - spp_si_mu * self.spp_si['quota']) * self._time_step
        #               = (molN/molC*day - 1/day * molN/molC) * day
        #               = molN/molC
        spp_si['biomass'] = self.spp_si['biomass'] + (spp_si_mu * self.spp_si['biomass']) * self._time_step
        #                 = (1/day * molC/cell) * day
        #                 = molC/cell
        spp_si['cell_size'] = allo.biomass_to_size(spp_si['biomass'])
        #                   = mu_m**3/cell
        spp_si['resource'] = R - resource_loss.sum() + resource_gains
        #        = molN/L - molN/L + molN/L
        #        = molN/L
        self.spp_si.update(spp_si)

    def _births(self):
        offspring = {}
        reproduce = self.spp_si['biomass'] > 2 * self.spp_si['ini_biomass']
        if any(reproduce):
            # print("Births!!")
            halfbiomass = self.spp_si['biomass'][reproduce] * 0.5
            halfquota = self.spp_si['quota'][reproduce] * self.spp_si['biomass'][reproduce] * 0.5 / halfbiomass
            halfcellsize = self.spp_si['cell_size'][reproduce] * 0.5
            self.spp_si['biomass'][reproduce] = halfbiomass  # set biomass to half biomass
            self.spp_si['quota'][reproduce] = halfquota  # set quota to half quota
            self.spp_si['cell_size'][reproduce] = halfcellsize  # set cell size to half cell size
            offspring['spp'] = self.spp_si['spp'][reproduce]
            offspring['ini_biomass'] = halfbiomass
            offspring['biomass'] = halfbiomass
            offspring['quota'] = halfquota
            offspring['cell_size'] = halfcellsize
            offspring['q_max'] = allo.q_max(halfcellsize) / halfbiomass
            offspring['q_min'] = allo.q_min(halfcellsize) / halfbiomass
            offspring['v_max'] = allo.v_max(halfcellsize) / halfbiomass
            offspring['mu_max'] = allo.mu_max(halfcellsize)
            offspring['kr'] = allo.k_r(halfcellsize)
            offspring['rep_nind'] = self.spp_si['rep_nind'][reproduce]

            # print({k: np.append(self.spp_si[k], offspring[k]) for k in offspring.keys()})
            self.spp_si.update({k: np.append(self.spp_si[k], offspring[k]) for k in offspring.keys()})
            reset_ids = np.zeros(self.spp_si['spp'].shape)
            for sp in self._params['spp_names']:
                reset_ids[np.char.count(self.spp_si['spp'], sp).nonzero()] = np.arange(
                    np.count_nonzero(np.char.count(self.spp_si['spp'], sp)))
            self.spp_si['id'] = reset_ids

    def _deaths(self):
        to_die = self._rng.random(size=self.spp_si['biomass'].size) < self._params['dilution_rate'] * self._time_step
        if any(to_die):
            # print("Deaths!!")
            to_live = to_die != True
            spp_si = {k: self.spp_si[k][to_live] for k in
                      ['spp', 'id', 'cell_size', 'q_max', 'q_min', 'v_max', 'mu_max',
                       'kr', 'ini_biomass', 'biomass', 'quota', 'rep_nind']}
            spp_si['time'] = self.spp_si['time']
            spp_si['resource'] = self.spp_si['resource']
            self.spp_si = spp_si

    def _split_combine(self):
        """
        Algorithm to control the number of super-individuals during the simulation
        by splitting or combining them when their numbers increase or decrease
        predefined limits (Clark et al. 2011).
        """
        _dtf_spp_si = pd.DataFrame(self.spp_si).groupby('spp')

        if any(_dtf_spp_si.spp.count() > self._params['nsi_max']):
            # print('combine!')
            _dd = _dtf_spp_si.apply(self._combine).to_dict(orient='list')
            _spp_si = {k: np.array(v) for k, v in _dd.items()}
            _spp_si['time'] = self.spp_si['time']
            _spp_si['resource'] = self.spp_si['resource']
            # print(_spp_si)
            self.spp_si.update(_spp_si)
        elif any(_dtf_spp_si.spp.count() < self._params['nsi_min']):
            # print('split!')
            _dd = _dtf_spp_si.apply(self._split).to_dict(orient='list')
            _spp_si = {k: np.array(v) for k, v in _dd.items()}
            _spp_si['time'] = self.spp_si['time']
            _spp_si['resource'] = self.spp_si['resource']
            # print(_spp_si)
            self.spp_si.update(_spp_si)
        else:
            # print('Do Nothing!')
            pass

    def _combine(self, dtf):
        if dtf.shape[0] > self._params['nsi_max']:
            # number of individuals to combine
            no_ind_comb = (dtf.shape[0] - self._params['nsi_max']) * 2
            # print('no_ind_comb', no_ind_comb)
            # labels for pair of individuals to combine
            to_combine = np.char.add('C', np.repeat(range(no_ind_comb // 2), 2).astype(str))
            # sorted and unsorted individuals based biomass and number of individuals to combine
            sort_ind = dtf.sort_values('biomass').iloc[-no_ind_comb:].assign(to_comb=to_combine)
            unsort_ind = dtf.sort_values('biomass').iloc[:-no_ind_comb]
            # calculation of biomass, cell_size and quota weighted means for the combine individuals
            new_biomass = sort_ind.groupby('to_comb').apply(
                lambda x: np.sum(x.biomass * x.rep_nind) / np.sum(x.rep_nind))
            new_cell_size = sort_ind.groupby('to_comb').apply(
                lambda x: np.sum(x.cell_size * x.rep_nind) / np.sum(x.rep_nind))
            new_quota = sort_ind.groupby('to_comb').apply(lambda x: np.sum(x.quota * x.rep_nind) / np.sum(x.rep_nind))
            new_rep_nind = sort_ind.groupby('to_comb').rep_nind.sum()
            new_q_max = allo.q_max(new_cell_size) / new_biomass
            new_q_min = allo.q_min(new_cell_size) / new_biomass
            new_v_max = allo.v_max(new_cell_size) / new_biomass
            new_mu_max = allo.mu_max(new_cell_size)
            new_kr = allo.k_r(new_cell_size)
            new_time = sort_ind.groupby('to_comb').time.first()
            new_resource = sort_ind.groupby('to_comb').resource.first()
            new_spp = sort_ind.groupby('to_comb').spp.first()

            new_dtf = (pd.concat([new_time, new_resource, new_spp,
                                  new_cell_size, new_q_max, new_q_min,
                                  new_v_max, new_mu_max, new_kr,
                                  new_biomass, new_biomass, new_quota, new_rep_nind], axis=1)
                       .reset_index()
                       .rename(columns={'to_comb': 'id',
                                        0: 'cell_size', 1: 'q_max', 2: 'q_min',
                                        3: 'v_max', 4: 'mu_max', 5: 'kr',
                                        6: 'ini_biomass', 7: 'biomass', 8: 'quota',
                                        'new_rep_nind': 'red_nind',
                                        })
                       )
            new_id = np.arange(unsort_ind.shape[0] + new_dtf.shape[0])

            return unsort_ind.append(new_dtf).assign(id=new_id).reset_index(drop=True)

        else:
            return dtf

    def _split(self, dtf):
        if dtf.shape[0] < self._params['nsi_min']:
            # number of individuals to split
            no_ind_split = self._params['nsi_max'] - dtf.shape[0]
            # labels for pair of individuals to combine
            # to_split = np.char.add('S', np.arange(no_ind_split).astype(str))
            # sorted and unsorted individuals based biomass and number of individuals to combine
            sort_ind = dtf.sort_values('biomass').iloc[:no_ind_split]  # .assign(to_split=to_split)
            unsort_ind = dtf.sort_values('biomass').iloc[no_ind_split:]
            # calculation of biomass, cell_size and quota weighted means for the split individuals
            sort_ind.rep_nind = sort_ind.rep_nind / 2
            new_dtf = sort_ind.append(sort_ind.copy())
            new_id = np.arange(unsort_ind.shape[0] + new_dtf.shape[0])
            return unsort_ind.append(new_dtf).assign(id=new_id).reset_index(drop=True)

        else:
            return dtf

    def to_DataFrame(self):
        return pd.DataFrame(self.spp_si).reset_index(drop=True)


class PhytoCell:
    """
    Phytoplankton cell (agent) is characterised by its nutrient related traits,
    which are determined by allometric functions observed by
    Marañon et al. (2013, Eco. Lett.)
    :param min_cell_size: minimum cell size of phytoplankton population (um^3)
    :param max_cell_size: maximum cell size of phytoplankton population (um^3)
    :param init_cell_size: initial cell size (um^3 / cell)
    :param init_biomass: initial biomass of cell (mol C / cell)
    :param init_quota: initial quota (mol N / mol C)
    """

    def __init__(self, min_cell_size=0.12, max_cell_size=2.5e7, rep_nind=None,
                 init_cell_size=None, init_biomass=None, init_quota=None,
                 init_size_random=True):
        """
        Phytoplankton cell (agent) is characterised by its nutrient related traits,
        which are determined by allometric functions observed by
        Marañon et al. (2013, Eco. Lett.)
        :param min_cell_size: float,
        minimum cell size of phytoplankton population (um^3)
        :param max_cell_size: float,
        maximum cell size of phytoplankton population (um^3)
        :param rep_nind: integer,
        abundance of individuals each super individual represents
        :param init_cell_size: float,
        initial cell size (um^3 / cell)
        :param init_biomass: float,
        initial biomass of cell (mol C / cell)
        :param init_quota: float,
        initial quota (mol N / mol C)
        :param init_size_random: bool

        """
        # traits and variables:
        if init_cell_size is None:
            if init_size_random:
                self.cell_size = 10 ** np.random.uniform(np.log10(min_cell_size),
                                                         np.log10(max_cell_size))  # Volume um3/cell
            else:
                self.cell_size = (min_cell_size + max_cell_size) / 2.
        else:
            self.cell_size = init_cell_size
        if init_biomass is None:
            self.ini_biomass = allo.biomass(self.cell_size)  # C biomass pgC/cell -> molC/cell
        else:
            self.ini_biomass = init_biomass  # molC/cell
        self.q_max = allo.q_max(
            self.cell_size) / self.ini_biomass  # Max Quota pgN/cell -> molN/molC
        self.q_min = allo.q_min(
            self.cell_size) / self.ini_biomass  # Min Quota pgN/cell -> molN/molC
        self.v_max = allo.v_max(
            self.cell_size) / self.ini_biomass  # Max uptake velocity pgN/cell*day -> molN/molC*day
        self.mu_max = allo.mu_max(self.cell_size)  # Maximum growth rate 1/day
        self.kr = allo.k_r(self.cell_size)  # resource half-saturation molN/L = MN
        if init_quota is None:
            self.quota = (self.q_min + self.q_max) / 2.  # Cell Quota (molN/molC)
        else:
            self.quota = init_quota  # Cell Quota (molN/molC)
        self.biomass = self.ini_biomass  # Cell biomass (molC/cell)
        self.V = 0.0
        self.mu = 0.0
        if rep_nind is None:
            self.rep_nind = 1e0
        else:
            self.rep_nind = rep_nind
        self.massbalance = None

    def growth(self, r, tstep):
        """
        Growth of phytoplankton cell following Droop's formulation (Droop 1965 J. Mar. bio. Asso. UK,
        Droop 1983 Bot. Mar.).
        :param r: float,
                  resource concentration MN
        :param tstep: float,
                      time step
        :return: updated biomass and cell size of individual PhytoCell
        """
        self.V = self.v_max * ((self.q_max - self.quota) / (self.q_max - self.q_min)) * (r / (self.kr + r))
        #      = molN/molC*day * ([molN/molC - molN/molC]/[molN/molC - molN/molC]) * (MN/[MN + MN])
        #      = molN/molC*day
        self.massbalance = self.V * self.biomass * self.rep_nind
        self.mu = self.mu_max * (1. - (self.q_min / self.quota))
        #       = 1/day * (1. - molN/molC / molN/molC)
        #       = 1/day
        self.quota += (self.V - self.mu * self.quota) * tstep
        #           = (molN/molC*day - 1/day * molN/molC) * day
        #           = molN/molC
        self.biomass += (self.mu * self.biomass) * tstep
        #             = (1/day * molC/cell) * day
        #             = molC/cell
        self.cell_size = allo.biomass_to_size(self.biomass)


class SBMi_asyn(SBMbase):
    """
    Size-based model of individual phytoplankton cells.
    :param nsi_spp: list or tuple of integers
        number of super individuals per species or strain
    :param nsi_min: integer
        minimum number of super individual for all species and strain
    :param nsi_max: integer
        maximum number of super individual for all species and strain
    :param time_end: integer
        final time in days used in the simulations
    :param time_step: float
        time steps used in the simulations
    :param print_time_step: integer
        time step in days used to store results
    :return: object with solution of size-based model
    """

    def __init__(self, ini_resource, ini_density, spp_names, min_cell_size, max_cell_size, dilution_rate, volume,
                 nsi_spp, nsi_min, nsi_max, time_end=20, time_step=1 / 24, print_time_step=1, random_seed=1234,
                 reproduction_random=False):
        super().__init__(ini_resource, ini_density, spp_names, min_cell_size, max_cell_size, dilution_rate, volume)

        self._rng = np.random.default_rng(random_seed)
        self._reproduction_random = reproduction_random
        # Simulation parameters
        self._params = {
            'ini_resource': ini_resource,
            'spp_names': spp_names,
            'ini_nind': ini_density,
            'min_cell_size': min_cell_size,
            'max_cell_size': max_cell_size,
            'nsi_spp': list(nsi_spp),
            'nsi_min': nsi_min,
            'nsi_max': nsi_max,
            'dilution_rate': dilution_rate,
            'volume': volume
        }
        self.dtf = pd.DataFrame({
            'time': None,
            'resource': None,
            'spp': np.array([]),
            'id': np.array([]),
            'cell_size': np.array([]),
            'q_max': np.array([]),
            'q_min': np.array([]),
            'v_max': np.array([]),
            'mu_max': np.array([]),
            'kr': np.array([]),
            'ini_biomass': np.array([]),
            'biomass': np.array([]),
            'quota': np.array([]),
            'rep_nind': np.array([])
        })
        self._R = ini_resource  # concentration of resource
        self._nsi_spp = list(nsi_spp)  # Number of super individuals per species
        self._dt = time_step  # time step of simulation fraction of day
        self._dtp = print_time_step  # time step to save results
        self._pdic = {}  # dictionary of phytoplankton agents
        self._ddic = {}  # dictionary of dead phytoplankton agents
        self._time_end = time_end
        self.run()

    def to_DataFrame(self, t_print):
        """
        Method to store result of the simulation into arrays
        :param t_print: integer
            time step to save results of simulation
        :return: instance arrays with saved results
        """
        abundance = []
        biomass = []
        quota = []
        spp = []
        ids = []
        cell_size = []
        q_max = []
        q_min = []
        v_max = []
        mu_max = []
        kr = []
        ini_biomass = []
        rep_nind = []

        for key in self._pdic.keys():
            abundance.append(self._pdic[key].rep_nind)
            biomass.append(self._pdic[key].biomass * self._pdic[key].rep_nind)
            quota.append(self._pdic[key].quota * self._pdic[key].biomass * self._pdic[key].rep_nind)
            spp.append(key[:2])
            ids.append(key[2:])
            cell_size.append(self._pdic[key].cell_size)
            q_max.append(self._pdic[key].q_max)
            q_min.append(self._pdic[key].q_min)
            v_max.append(self._pdic[key].q_max)
            mu_max.append(self._pdic[key].mu_max)
            kr.append(self._pdic[key].kr)
            ini_biomass.append(self._pdic[key].ini_biomass)
            rep_nind.append(self._pdic[key].rep_nind)

        _spp_si = {
            'time': t_print,
            'resource': self._R,
            'spp': spp,
            'id': ids,
            'cell_size': cell_size,
            'q_max': q_max,
            'q_min': q_min,
            'v_max': v_max,
            'mu_max': mu_max,
            'kr': kr,
            'ini_biomass': ini_biomass,
            'biomass': biomass,
            'quota': quota,
            'rep_nind': rep_nind
        }

        return pd.DataFrame(_spp_si).reset_index(drop=True)

    def _split_combine(self):
        """
        Algorithm to control the number of super-individuals during the simulation
        by splitting or combining them when their numbers increase or decrease
        predefined limits (Clark et al. 2011).
        """
        _nsi_ave = (self._params['nsi_min'] + self._params['nsi_max']) // 2
        while np.sum(self._nsi_spp) > self._params['nsi_max']:
            pdic_copy = self._pdic.copy()  # copy of pdic
            sp_idx = self._nsi_spp.index(np.max(self._nsi_spp))  # get index of sp with max number of super individuals
            sp_name = self._params['spp_names'][sp_idx]  # get name of sp with max number of super individuals
            while np.sum(self._nsi_spp) > _nsi_ave and self._nsi_spp[sp_idx] > 1:
                # dictionary filtered by species with max number of super individuals
                spdic = {key: val for key, val in pdic_copy.items() if key.startswith(sp_name)}
                # sorted keys of dictionary based on biomass, where first element in list is the smallest cell
                keysort = [key for key, val in sorted(spdic.items(), key=lambda kv: kv[1].biomass)]
                bmcombined = (pdic_copy[keysort[0]].biomass * pdic_copy[keysort[0]].rep_nind +
                              pdic_copy[keysort[1]].biomass * pdic_copy[keysort[1]].rep_nind) \
                             / (pdic_copy[keysort[0]].rep_nind + pdic_copy[keysort[1]].rep_nind)
                cscombined = (pdic_copy[keysort[0]].cell_size * pdic_copy[keysort[0]].rep_nind +
                              pdic_copy[keysort[1]].cell_size * pdic_copy[keysort[1]].rep_nind) \
                             / (pdic_copy[keysort[0]].rep_nind + pdic_copy[keysort[1]].rep_nind)
                qcombined = (pdic_copy[keysort[0]].quota * pdic_copy[keysort[0]].biomass * pdic_copy[
                    keysort[0]].rep_nind +
                             pdic_copy[keysort[1]].quota * pdic_copy[keysort[1]].biomass * pdic_copy[
                                 keysort[1]].rep_nind) \
                            / (pdic_copy[keysort[0]].rep_nind + pdic_copy[keysort[1]].rep_nind)
                qcombined = qcombined / bmcombined
                rncombined = pdic_copy[keysort[0]].rep_nind + pdic_copy[keysort[1]].rep_nind
                combinedkey = sp_name + str(len(keysort) - 2).zfill(5)
                newspdic = {sp_name + str(nag).zfill(5): pdic_copy[key] for nag, key in
                            zip(range(len(keysort) - 2), keysort[2:])}
                newspdic.update({combinedkey: PhytoCell(init_biomass=bmcombined, init_quota=qcombined,
                                                        init_cell_size=cscombined, rep_nind=rncombined)})
                newspdic.update({key: val for key, val in pdic_copy.items() if not sp_name in key})
                pdic_copy = newspdic.copy()
                self._nsi_spp[sp_idx] -= 1
            self._pdic = pdic_copy
        while np.sum(self._nsi_spp) < self._params['nsi_min']:
            # print('split')
            pdic_copy = self._pdic.copy()  # copy of pdic
            sp_idx = self._nsi_spp.index(min(self._nsi_spp))  # get index of sp with min number of super individuals
            sp_name = self._params['spp_names'][sp_idx]  # get name of sp with min number of super individuals
            while np.sum(self._nsi_spp) < _nsi_ave:
                # dictionary filtered by species with min number of super individuals
                spdic = {key: val for key, val in pdic_copy.items() if key.startswith(sp_name)}
                # sorted keys of dictionary based on biomass, where first element in list is the largest cell
                keysort = [key for key, val in sorted(spdic.items(), key=lambda kv: kv[1].biomass, reverse=True)]
                splitkey = sp_name + '_split'
                rnsplit = spdic[keysort[0]].rep_nind / 2
                spdic.update({splitkey: PhytoCell(init_biomass=spdic[keysort[0]].biomass,
                                                  init_cell_size=spdic[keysort[0]].cell_size,
                                                  init_quota=spdic[keysort[0]].quota * spdic[keysort[0]].biomass \
                                                             * spdic[keysort[0]].rep_nind,
                                                  rep_nind=rnsplit)})
                spdic[keysort[0]].rep_nind = rnsplit
                newspdic = {sp_name + str(nag).zfill(5): spdic[key] for nag, key in
                            zip(range(len(spdic)), spdic.keys())}
                newspdic.update({key: val for key, val in pdic_copy.items() if not sp_name in key})
                pdic_copy = newspdic.copy()
                self._nsi_spp[sp_idx] += 1
            self._pdic = pdic_copy

    def initialize(self):
        """
        Method to initialize phytoplankton cell agents
        :return: updated dictionary and arrays
        """
        self._pdic.update(
            {self._params['spp_names'][sp] + str(numag).zfill(5): PhytoCell(
                min_cell_size=self._params['min_cell_size'][sp],
                max_cell_size=self._params['max_cell_size'][sp],
                rep_nind=self._params['ini_nind'][sp] /
                         self._params['nsi_spp'][sp])
                for sp in range(len(self._params['spp_names'])) for numag in
                range(self._nsi_spp[sp])})  # add phytoplankton agents

    def update(self):
        """
        Method to asynchronous update the state of all agent and resource
        :return: updated agents and resource
        """
        pkeys = list(self._pdic.keys())
        pdead = {}
        # Computations for all agents
        for i in range(len(pkeys)):
            randkey = self._rng.choice(pkeys)
            pkeys.remove(randkey)
            self._pdic[randkey].growth(self._R, self._dt)  # growth of a single phytoplankton agent
            spp_idx = self._params['spp_names'].index(randkey[0:2])
            self._R -= self._pdic[randkey].massbalance * self._dt / self._params['volume']  # loss of R due to uptake
            #       = molN/molC*hour * molC/cell * cell/agent * hour/L
            #       = molN/L
            if self._reproduction_random:  # Random reproduction
                if self._rng.random() < self._pdic[randkey].mu * self._dt:
                    halfbiomass = self._pdic[randkey].biomass * 0.5
                    halfquota = self._pdic[randkey].quota * self._pdic[randkey].biomass * 0.5 / halfbiomass
                    halfcellsize = self._pdic[randkey].cell_size * 0.5
                    self._pdic[randkey].biomass = halfbiomass  # set biomass to half biomass
                    self._pdic[randkey].quota = halfquota  # set quota to half quota
                    self._pdic[randkey].cell_size = halfcellsize  # set cell size to half cell size
                    maxpalive = np.max([int(key[2:]) for key in self._pdic.keys() if randkey[0:2] in key])
                    newpcellkey = randkey[0:2] + str(maxpalive + 1).zfill(5)  # key of new phytoplankton cell
                    # create new phytoplankton with half cell size, biomass and quota
                    self._pdic.update({newpcellkey: PhytoCell(init_biomass=halfbiomass, init_cell_size=halfcellsize,
                                                              rep_nind=self._pdic[randkey].rep_nind,
                                                              init_quota=halfquota)})  #
                    self._nsi_spp[spp_idx] += 1
            else:  # Deterministic cell size reproduction
                if self._pdic[randkey].biomass > 2 * self._pdic[randkey].ini_biomass:
                    halfbiomass = self._pdic[randkey].biomass * 0.5
                    halfquota = self._pdic[randkey].quota * self._pdic[randkey].biomass * 0.5 / halfbiomass
                    halfcellsize = self._pdic[randkey].cell_size * 0.5
                    self._pdic[randkey].biomass = halfbiomass  # set biomass to half biomass
                    self._pdic[randkey].quota = halfquota  # set quota to half quota
                    self._pdic[randkey].cell_size = halfcellsize  # set cell size to half cell size
                    maxpalive = np.max([int(key[2:]) for key in self._pdic.keys() if randkey[0:2] in key])
                    newpcellkey = randkey[0:2] + str(maxpalive + 1).zfill(5)  # key of new phytoplankton cell
                    # create new phytoplankton with half cell size, biomass and quota
                    self._pdic.update({newpcellkey: PhytoCell(init_biomass=halfbiomass, init_cell_size=halfcellsize,
                                                              rep_nind=self._pdic[randkey].rep_nind,
                                                              init_quota=halfquota)})  #
                    self._nsi_spp[spp_idx] += 1
            if self._rng.random() < self._params[
                'dilution_rate'] * self._dt:  # Mortality based on probability of being washout
                maxpdead = len(pdead)
                pdeadkey = 'd' + randkey[0:2] + str(maxpdead + 1).zfill(5)
                pdead.update({pdeadkey: self._pdic[randkey]})
                self._pdic.pop(randkey)
                self._nsi_spp[spp_idx] -= 1

        self._split_combine()

    def run(self):
        """
        Method to execute the simulation
        :return: simulation results
        """
        simtime = self._dt  # Simulation time
        printtime = self._dtp  # Print time step
        indexpt = 1  # index of print time step to save results
        self.initialize()
        #  Save initial conditions into output dataframe
        self.dtf = self.dtf.append(self.to_DataFrame(0))
        while simtime <= self._time_end + self._dt:
            self._R += (self._params['dilution_rate'] * (
                    self._params['ini_resource'] - self._R)) * self._dt  # Change in Resource concentration
            self.update()  # Update state of all agents
            if simtime >= printtime:  # Save results
                self.dtf = self.dtf.append(self.to_DataFrame(printtime))
                printtime += self._dtp
                indexpt += 1
            simtime += self._dt  # increase simulation time by one time step (fraction of a day)
