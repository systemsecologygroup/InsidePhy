import numpy as np
import random
import time
import insidephy.size_based_models.allometric_functions as AlloFunc


class PhytoCell:
    """
    Phytoplankton cell (agent) is characterised by its nutrient related traits,
    which are determined by allometric functions observed by
    Marañon et al. (2013, Eco. Lett.)
    :param min_cell_size: minimum cell size of phytoplankton population (um^3)
    :param max_cell_size: maximum cell size of phytoplankton population (um^3)
    :param init_cell_size: initial cell size (um^3 / cell)
    :param init_biomass: initial biomass of cell (mol C / cell)
    :param init_quota: initial quota (umol N / mol C)
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
            self.ini_biomass = AlloFunc.biomass(self.cell_size)  # C biomass pgC/cell -> molC/cell
        else:
            self.ini_biomass = init_biomass  # molC/cell
        self.q_max = AlloFunc.q_max(
            self.cell_size) / self.ini_biomass  # Max Quota pgN/cell -> molN/molC
        self.q_min = AlloFunc.q_min(
            self.cell_size) / self.ini_biomass  # Min Quota pgN/cell -> molN/molC
        self.v_max = AlloFunc.v_max(
            self.cell_size) / self.ini_biomass  # Max uptake velocity pgN/cell*day -> molN/molC*day
        self.mu_max = AlloFunc.mu_max(self.cell_size)  # Maximum growth rate 1/day
        self.kr = AlloFunc.k_r(self.cell_size)  # resource half-saturation molN/L = MN
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
        self.V = self.v_max * ((self.q_max - self.quota) / (self.q_max - self.q_min)) * (r / (self.kr + r))
        #      = molN/molC*day * ([molN/molC - molN/molC]/[molN/molC - molN/molC]) * (MN/[MN + MN])
        #      = molN/molC*day
        self.massbalance = self.V * self.biomass * self.rep_nind
        self.mu = self.mu_max * (1. - (self.q_min / self.quota))  # 0.
        #       = 1/day * (1. - molN/molC / molN/molC)
        #       = 1/day
        self.quota += (self.V - self.mu * self.quota) * tstep
        #           = (molN/molC*day - 1/day * molN/molC) * day
        #           = molN/molC
        self.biomass += (self.mu * self.biomass) * tstep
        #             = (1/day * molC/cell) * day
        #             = molC/cell
        self.cell_size = AlloFunc.biomass_to_size(self.biomass)


class SBMi:
    """
    Size-based model of individual phytoplankton cells.
    :return: object with solution of size-based model
    """

    def __init__(self, ini_resource, ini_density, spp_names, min_size, max_size,
                 nsi_spp, nsi_min, nsi_max, dilution_rate,
                 volume, time_end=20, time_step=1/24, print_time_step=1,
                 timeit=False):
        if not all([isinstance(lst, list) for lst in iter([spp_names, ini_density, min_size, max_size, nsi_spp])]):
            raise TypeError('Error on input parameters spp_names, ini_density, min_size, max_size or nsi_spp. '
                            'They must be type list.')
        if not all([len(lst) == len(spp_names) for lst in iter([spp_names, ini_density, min_size, max_size, nsi_spp])]):
            raise ValueError("initial values of spp_names, ini_density, min_size, max_size and nsi_spp "
                             "must be lists of the same length depending on the number of species use "
                             "in the simulation")

        self.timeit = timeit
        self.R0 = ini_resource  # initial concentration of resource
        self.R = ini_resource  # concentration of resource

        self.spp_names = spp_names
        self.nind = ini_density  # initial number of individuals
        self.minsize = min_size  # Minimum cell size in um^3
        self.maxsize = max_size  # Maximum cell size in um^3
        self.nsi_min = nsi_min  # Minimum number of super individuals per species
        self.nsi_max = nsi_max  # Maximum number of super individuals per species
        self.nsi_ave = (nsi_min + nsi_max) // 2  # Average number of super individuals
        self.nsi_spp = nsi_spp  # Number of super individuals per species
        self.dt = time_step  # time step of simulation fraction of day
        self.dtp = print_time_step  # time step to print
        self.time = np.linspace(0, time_end, int((time_end + self.dtp) / self.dtp))
        self.dilution_rate = dilution_rate  # dilution rate
        self.volume = volume  # volume
        self.pdic = {}  # dictionary of phytoplankton agents
        self.ddic = {}  # dictionary of dead phytoplankton agents
        # initialize output arrays
        self.resource = np.zeros(len(self.time))
        self.abundance = np.zeros(len(self.time))
        self.biomass = np.zeros(len(self.time))
        self.quota = np.zeros(len(self.time))
        self.massbalance = np.zeros(len(self.time))
        self.number_si = np.zeros((len(self.spp_names), len(self.time)))
        self.agents_size = np.zeros((len(self.spp_names), len(self.time), nsi_max))
        self.agents_biomass = np.zeros((len(self.spp_names), len(self.time), nsi_max))
        self.agents_abundance = np.zeros((len(self.spp_names), len(self.time), nsi_max))
        self.agents_growth = np.zeros((len(self.spp_names), len(self.time), nsi_max))
        self.agents_size[:] = np.nan
        self.agents_biomass[:] = np.nan
        self.agents_abundance[:] = np.nan
        self.agents_growth[:] = np.nan
        np.random.seed(1234)
        self.initialize()
        self.run()

    def save_to_array(self, t_print):
        # Resource concentration
        # self.resource[t_print] = self.R
        abundance = []
        biomass = []
        quota = []

        for key in self.pdic.keys():
            abundance.append(self.pdic[key].rep_nind)
            biomass.append(self.pdic[key].biomass * self.pdic[key].rep_nind)
            quota.append(self.pdic[key].quota * self.pdic[key].biomass * self.pdic[key].rep_nind)

        quota_deadcell = [self.ddic[key].quota * self.ddic[key].biomass * self.ddic[key].rep_nind for key in
                          self.ddic.keys()]

        # Resource concentration
        self.resource[t_print] = self.R
        # Abundance
        self.abundance[t_print] = np.sum(abundance)
        # total biomass of phytoplankton agents
        self.biomass[t_print] = np.sum(biomass)
        # PON of phytoplankton agents
        self.quota[t_print] = np.sum(quota)
        # Mass Balance
        self.massbalance[t_print] = self.R + 1. / self.volume * np.sum(quota) + np.sum(quota_deadcell)
        # Number of super individuals
        self.number_si[:, t_print] = self.nsi_spp

        # Agents
        for i, spp in enumerate(self.spp_names):
            for k, key in enumerate(self.pdic.keys()):
                if spp in key:
                    # print(i, spp, k, key)
                    # Cell size
                    self.agents_size[i, t_print, k] = self.pdic[key].cell_size
                    # Biomass
                    self.agents_biomass[i, t_print, k] = self.pdic[key].biomass * self.pdic[key].rep_nind
                    # Abundance
                    self.agents_abundance[i, t_print, k] = self.pdic[key].rep_nind
                    # Growth rate
                    self.agents_growth[i, t_print, k] = self.pdic[key].mu

    def split_combine(self):
        """
        Algorithm to control the number of super-individuals during the simulation
        by splitting or combining them when their numbers increase or decrease
        predefined limits (Clark et al. 2011).
        """
        while np.sum(self.nsi_spp) > self.nsi_max:
            pdic_copy = self.pdic.copy()  # copy of pdic
            sp_idx = self.nsi_spp.index(np.max(self.nsi_spp))  # get index of sp with max number of super individuals
            sp_name = self.spp_names[sp_idx]  # get name of sp with max number of super individuals
            while np.sum(self.nsi_spp) > self.nsi_ave and self.nsi_spp[sp_idx] > 1:
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
                self.nsi_spp[sp_idx] -= 1
            self.pdic = pdic_copy
        while np.sum(self.nsi_spp) < self.nsi_min:
            # print('split')
            pdic_copy = self.pdic.copy()  # copy of pdic
            sp_idx = self.nsi_spp.index(min(self.nsi_spp))  # get index of sp with min number of super individuals
            sp_name = self.spp_names[sp_idx]  # get name of sp with min number of super individuals
            while np.sum(self.nsi_spp) < self.nsi_ave:
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
                self.nsi_spp[sp_idx] += 1
            self.pdic = pdic_copy

    def initialize(self):
        self.pdic.update({self.spp_names[sp] + str(numag).zfill(5): PhytoCell(min_cell_size=self.minsize[sp],
                                                                              max_cell_size=self.maxsize[sp],
                                                                              rep_nind=self.nind[sp] / self.nsi_spp[sp])
                          for sp in range(len(self.spp_names)) for numag in
                          range(self.nsi_spp[sp])})  # add phytoplankton agents
        #  Save initial conditions into output arrays
        self.save_to_array(0)

    def update(self):
        # prand = self.pdic.copy()
        pkeys = list(self.pdic.keys())
        pdead = {}
        # palive = self.pdic.copy()
        # Computations for all agents
        # for i in range(len(prand)):  # - 1
        for i in range(len(pkeys)):  # - 1
            # randidx = random.randint(0, len(prand) - 1)
            # randkey = list(prand.keys())[randidx]
            # prand.pop(randkey)
            randidx = random.randint(0, len(pkeys) - 1)
            randkey = pkeys[randidx]
            pkeys.remove(randkey)
            self.pdic[randkey].growth(self.R, self.dt)  # growth of a single phytoplankton agent
            spp_idx = self.spp_names.index(randkey[0:2])
            self.R -= self.pdic[randkey].massbalance * self.dt / self.volume  # loss of R due to uptake
            #       = molN/molC*hour * molC/cell * cell/agent * hour/L
            #       = molN/L
            if self.pdic[randkey].biomass > 2 * self.pdic[randkey].ini_biomass:  # Reproduction
                halfbiomass = self.pdic[randkey].biomass * 0.5
                halfquota = self.pdic[randkey].quota * self.pdic[randkey].biomass * 0.5 / halfbiomass
                halfcellsize = self.pdic[randkey].cell_size * 0.5
                self.pdic[randkey].biomass = halfbiomass  # set biomass to half biomass
                self.pdic[randkey].quota = halfquota  # set quota to half quota
                self.pdic[randkey].cell_size = halfcellsize  # set cell size to half cell size
                maxpalive = np.max([int(key[2:]) for key in self.pdic.keys() if randkey[0:2] in key])
                newpcellkey = randkey[0:2] + str(maxpalive + 1).zfill(5)  # key of new phytoplankton cell
                # create new phytoplankton with half cell size, biomass and quota
                self.pdic.update({newpcellkey: PhytoCell(init_biomass=halfbiomass, init_cell_size=halfcellsize,
                                                         rep_nind=self.pdic[randkey].rep_nind,
                                                         init_quota=halfquota)})  #
                self.nsi_spp[spp_idx] += 1
            # L = (1. - (self.pdic[randkey].q_min / self.pdic[randkey].quota))  # N limitation term
            mu_D = self.dilution_rate  # * (1. - L)  # mortality rate 1/day
            if random.random() < mu_D * self.dt:  # Mortality based on probability based on dilution
                maxpdead = len(pdead)
                pdeadkey = 'd' + randkey[0:2] + str(maxpdead + 1).zfill(5)
                pdead.update({pdeadkey: self.pdic[randkey]})
                self.pdic.pop(randkey)
                self.nsi_spp[spp_idx] -= 1

        # Update dictionary of dead and alive phytoplankton agents
        # self.pdic = {}
        # for spp in self.spp_names:
        #     len_spp_nsi = len([key for key in palive.keys() if spp in key])
        #     self.pdic.update({spp + str(nag).zfill(5): palive[kk] for nag, kk in
        #                       zip(range(len_spp_nsi), palive.keys())})
        # numpdead = len(self.ddic)
        # self.ddic.update({kk[0:3] + str(numpdead + nag).zfill(5): pdead[kk] for nag, kk in
        #                   zip(range(len(pdead)), pdead.keys())})

        # resampling of super individuals
        self.split_combine()

    def run(self):
        start_comp_time = time.time()
        simtime = self.dt  # Simulation time
        printtime = self.dtp  # Print time step
        indexpt = 1  # index of print time step to save results
        while simtime <= self.time.max() + self.dt:
            self.R += (self.dilution_rate * (self.R0 - self.R)) * self.dt  # Change in Resource concentration
            self.update()  # Update state of all agents
            if simtime >= printtime:  # Save results
                self.save_to_array(printtime)
                printtime += self.dtp
                indexpt += 1
            simtime += self.dt  # increase simulation time by one time step (fraction of a day)
        if self.timeit:
            print('SBMi total simulation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))
