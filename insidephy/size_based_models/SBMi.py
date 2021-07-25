import numpy as np
import random
import time
import insidephy.size_based_models.allometric_functions as allo


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
                 init_cell_size=None, init_biomass=None, init_quota=None):
        # traits and variables:
        if init_cell_size is None:
            self.cell_size = 10 ** np.random.uniform(np.log10(min_cell_size),
                                                     np.log10(max_cell_size))  # Volume um3/cell
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


class SBMi:
    """
    Size-based model of individual phytoplankton cells.
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
    :param nsi_spp: list or tuple of integers
        number of super individuals per species or strain
    :param nsi_min: integer
        minimum number of super individual for all species and strain
    :param nsi_max: integer
        maximum number of super individual for all species and strain
    :param dilution_rate: float
        rate of medium exchange in the culture system
    :param volume: float
        volume of the culture system
    :param time_end: integer
        final time in days used in the simulations
    :param time_step: float
        time steps used in the simulations
    :param print_time_step: integer
        time step in days used to store results
    :param timeit: bool
        time of the simulation
    :return: object with solution of size-based model
    """

    def __init__(self, ini_resource, ini_density, spp_names, min_size, max_size,
                 nsi_spp, nsi_min, nsi_max, dilution_rate, volume,
                 time_end=20, time_step=1 / 24, print_time_step=1,
                 timeit=False):
        if not all([isinstance(item, list) or isinstance(item, tuple)
                    for item in iter([spp_names, ini_density, min_size, max_size, nsi_spp])]):
            raise TypeError('Error on input parameters spp_names, ini_density, min_size, max_size or nsi_spp. '
                            'Input parameters must be type list or tuple.')
        if not all([len(lst) == len(spp_names) for lst in iter([spp_names, ini_density, min_size, max_size, nsi_spp])]):
            raise ValueError("initial values of spp_names, ini_density, min_size, max_size and nsi_spp "
                             "must be lists of the same length depending on the number of species use "
                             "in the simulation")

        # Simulation parameters and variables
        self._r0 = ini_resource  # initial concentration of resource
        self._spp_names = spp_names  # Two letter code name for species or strains
        self._nind = ini_density  # initial number of individuals
        self._minsize = min_size  # Minimum cell size in um^3
        self._maxsize = max_size  # Maximum cell size in um^3
        self._nsi_min = nsi_min  # Minimum number of super individuals per species or strains
        self._nsi_max = nsi_max  # Maximum number of super individuals per species or strains
        self._dilution_rate = dilution_rate  # dilution rate
        self._volume = volume  # volume
        self._timeit = timeit
        self._R = ini_resource  # concentration of resource
        self._nsi_spp = list(nsi_spp)  # Number of super individuals per species
        self._nsi_ave = (nsi_min + nsi_max) // 2  # Average number of super individuals
        self._dt = time_step  # time step of simulation fraction of day
        self._dtp = print_time_step  # time step to save results
        self._pdic = {}  # dictionary of phytoplankton agents
        self._ddic = {}  # dictionary of dead phytoplankton agents
        # initialize output arrays
        self.time = np.linspace(0, time_end, int((time_end + self._dtp) / self._dtp))
        self.resource = np.zeros(len(self.time))
        self.abundance = np.zeros(len(self.time))
        self.biomass = np.zeros(len(self.time))
        self.quota = np.zeros(len(self.time))
        self.massbalance = np.zeros(len(self.time))
        self.number_si = np.zeros((len(self._spp_names), len(self.time)))
        self.agents_size = np.zeros((len(self._spp_names), len(self.time), nsi_max))
        self.agents_biomass = np.zeros((len(self._spp_names), len(self.time), nsi_max))
        self.agents_abundance = np.zeros((len(self._spp_names), len(self.time), nsi_max))
        self.agents_growth = np.zeros((len(self._spp_names), len(self.time), nsi_max))
        self.agents_quota = np.zeros((len(self._spp_names), len(self.time), nsi_max))
        self.agents_size[:] = np.nan
        self.agents_biomass[:] = np.nan
        self.agents_abundance[:] = np.nan
        self.agents_growth[:] = np.nan
        self.agents_quota[:] = np.nan
        np.random.seed(1234)
        self._initialize()
        self._run()

    def _save_to_array(self, t_print):
        """
        Method to store result of the simulation into arrays
        :param t_print: integer
            time step to save results of simulation
        :return: instance arrays with saved results
        """
        abundance = []
        biomass = []
        quota = []

        for key in self._pdic.keys():
            abundance.append(self._pdic[key].rep_nind)
            biomass.append(self._pdic[key].biomass * self._pdic[key].rep_nind)
            quota.append(self._pdic[key].quota * self._pdic[key].biomass * self._pdic[key].rep_nind)

        quota_deadcell = [self._ddic[key].quota * self._ddic[key].biomass * self._ddic[key].rep_nind for key in
                          self._ddic.keys()]

        # Resource concentration
        self.resource[t_print] = self._R
        # Abundance
        self.abundance[t_print] = np.sum(abundance)
        # total biomass of phytoplankton agents
        self.biomass[t_print] = np.sum(biomass)
        # PON of phytoplankton agents
        self.quota[t_print] = np.sum(quota)
        # Mass Balance
        self.massbalance[t_print] = self._R + 1. / self._volume * np.sum(quota) + np.sum(quota_deadcell)
        # Number of super individuals
        self.number_si[:, t_print] = self._nsi_spp

        # Agents
        for i, spp in enumerate(self._spp_names):
            for k, key in enumerate(self._pdic.keys()):
                if spp in key:
                    # Cell size
                    self.agents_size[i, t_print, k] = self._pdic[key].cell_size
                    # Biomass
                    self.agents_biomass[i, t_print, k] = self._pdic[key].biomass * self._pdic[key].rep_nind
                    # Abundance
                    self.agents_abundance[i, t_print, k] = self._pdic[key].rep_nind
                    # Growth rate
                    self.agents_growth[i, t_print, k] = self._pdic[key].mu
                    # Quota
                    self.agents_quota[i, t_print, k] = self._pdic[key].quota * self._pdic[key].biomass * \
                                                       self._pdic[key].rep_nind

    def _split_combine(self):
        """
        Algorithm to control the number of super-individuals during the simulation
        by splitting or combining them when their numbers increase or decrease
        predefined limits (Clark et al. 2011).
        """
        while np.sum(self._nsi_spp) > self._nsi_max:
            pdic_copy = self._pdic.copy()  # copy of pdic
            sp_idx = self._nsi_spp.index(np.max(self._nsi_spp))  # get index of sp with max number of super individuals
            sp_name = self._spp_names[sp_idx]  # get name of sp with max number of super individuals
            while np.sum(self._nsi_spp) > self._nsi_ave and self._nsi_spp[sp_idx] > 1:
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
        while np.sum(self._nsi_spp) < self._nsi_min:
            # print('split')
            pdic_copy = self._pdic.copy()  # copy of pdic
            sp_idx = self._nsi_spp.index(min(self._nsi_spp))  # get index of sp with min number of super individuals
            sp_name = self._spp_names[sp_idx]  # get name of sp with min number of super individuals
            while np.sum(self._nsi_spp) < self._nsi_ave:
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

    def _initialize(self):
        """
        Method to initialize phytoplankton cell agents
        :return: updated dictionary and arrays
        """
        self._pdic.update({self._spp_names[sp] + str(numag).zfill(5): PhytoCell(min_cell_size=self._minsize[sp],
                                                                                max_cell_size=self._maxsize[sp],
                                                                                rep_nind=self._nind[sp] / self._nsi_spp[sp])
                           for sp in range(len(self._spp_names)) for numag in
                           range(self._nsi_spp[sp])})  # add phytoplankton agents
        #  Save initial conditions into output arrays
        self._save_to_array(0)

    def _update(self):
        """
        Method to update the state of all agent and resource
        :return: updated agents and resource
        """
        pkeys = list(self._pdic.keys())
        pdead = {}
        # Computations for all agents
        for i in range(len(pkeys)):
            randidx = random.randint(0, len(pkeys) - 1)
            randkey = pkeys[randidx]
            pkeys.remove(randkey)
            self._pdic[randkey].growth(self._R, self._dt)  # growth of a single phytoplankton agent
            spp_idx = self._spp_names.index(randkey[0:2])
            self._R -= self._pdic[randkey].massbalance * self._dt / self._volume  # loss of R due to uptake
            #       = molN/molC*hour * molC/cell * cell/agent * hour/L
            #       = molN/L
            if self._pdic[randkey].biomass > 2 * self._pdic[randkey].ini_biomass:  # Reproduction
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
            if random.random() < self._dilution_rate * self._dt:  # Mortality based on probability of being washout
                maxpdead = len(pdead)
                pdeadkey = 'd' + randkey[0:2] + str(maxpdead + 1).zfill(5)
                pdead.update({pdeadkey: self._pdic[randkey]})
                self._pdic.pop(randkey)
                self._nsi_spp[spp_idx] -= 1

        self._split_combine()

    def _run(self):
        """
        Method to execute the simulation
        :return: simulation results
        """
        start_comp_time = time.time()
        simtime = self._dt  # Simulation time
        printtime = self._dtp  # Print time step
        indexpt = 1  # index of print time step to save results
        while simtime <= self.time.max() + self._dt:
            self._R += (self._dilution_rate * (self._r0 - self._R)) * self._dt  # Change in Resource concentration
            self._update()  # Update state of all agents
            if simtime >= printtime:  # Save results
                self._save_to_array(printtime)
                printtime += self._dtp
                indexpt += 1
            simtime += self._dt  # increase simulation time by one time step (fraction of a day)
        if self._timeit:
            print('SBMi total simulation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))
