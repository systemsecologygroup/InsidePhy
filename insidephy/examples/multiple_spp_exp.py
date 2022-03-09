import time
from datetime import datetime
from re import search
import pkg_resources
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from itertools import combinations
# from insidephy.size_based_models.SBMc import SBMc
# from insidephy.size_based_models.SBMi import SBMi
from insidephy.size_based_models.sbm import SBMc, SBMi_asyn, SBMi_syn
import dask
from dask.distributed import Client, LocalCluster, SSHCluster
from toolz import partition_all


def run_sims_delayed(num_spp_exp, rel_size_range=0.25, dilution=0.0, volume=1.0, time_end=20,
                     sbmc_numsc=[10], sbmi_nsispp=[101], sbmi_nsimin=100, sbmi_nsimax=1000, sbmi_ts=1 / 24,
                     ssh=False, ssh_username=None, ssh_pw=None, nprocs=4, nthreads=1, mem_lim=2e9,
                     compute_batches=1000, sbmi_variant=True, sbmc_variant=True):
    """
    function to calculate multiple species experiments from 22 phytoplankton species data
    reported in Marañón et al. (2013 Eco. Lett.). Notice that number of simulations
    and therefore execution time, memory, and hard drive space grows exponentially
    to a maximum around half the number of species.
    :param num_spp_exp: integer
        number of species to combine on each experiment from a minimum of 1 (similar to single species experiments) to
          a maximum of 22 (a community composed of maximum number of species investigated by Marañón et al.).
    :param rel_size_range: float
        initial size variability relative to its initial mean size
    :param dilution: float
        rate of medium exchange in the culture system
    :param volume: float
        volume of the culture system
    :param sbmc_numsc: list or tuple of integers
        number of size classes per species used in SBMc
    :param time_end: integer
        final time in days used in the simulations
    :param sbmi_nsispp: list or tuple of integers
        number of super individuals per species
    :param sbmi_nsimin: integer
        minimum number of super individual for all species
    :param sbmi_nsimax: integer
        maximum number of super individual for all species
    :param sbmi_ts: float
        time steps used in the simulations
    :param ssh: bool
        whether to use ssh to forward diagnostics of cluster
        see documentation of dask.distributed.SSHCluster
    :param ssh_username: string
        user name to connect to cluster via ssh
        see documentation of dask.distributed.SSHCluster
    :param ssh_pw:  string
        password to connect to cluster via ssh
        see documentation of dask.distributed.SSHCluster
    :param n_procs: integer
        number of cpus to compute the simulations
    :param n_threads:
        number of threads per cpu to use in the computations
    :param mem_lim:
        maximum memory usage per cpu
    :param compute_batches: integer
        number of simulations to compute include in a batch for parellel computation
    :param sbmi_variant: bool
        whether to calculate a model based on individual SBMi
    :param sbmc_variant: bool
        whether to calculate a model based on size classes SBMc
    :return: ncfiles with restuls of simulations
    """

    start_comp_time = time.time()
    start_datetime = datetime.now()
    print('Start of experiments for combinations of %.i species on' % num_spp_exp,
          start_datetime.strftime("%b-%d-%Y %H:%M:%S"))

    # try:
    #     client = Client('tcp://localhost:8786', timeout=1)
    # except OSError:
    if ssh:
        cluster = SSHCluster(["localhost", "localhost", "localhost", "localhost"],
                             connect_options={"known_hosts": None,
                                              "username": ssh_username,
                                              "password": ssh_pw},
                             worker_options={"nprocs": nprocs, "nthreads": nthreads, "memory_limit": mem_lim},
                             scheduler_options={"port": 0, "dashboard_address": ":8787"})
        client = Client(cluster)
    else:
        cluster = LocalCluster(n_workers=nprocs, threads_per_worker=nthreads, memory_limit=mem_lim,
                               scheduler_port=8786)
        client = Client(cluster)

    def get_init_param(exp_num):
        data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
        allometries = pd.read_hdf(data_path, 'allodtf')
        bath_cultures = pd.read_hdf(data_path, 'batchdtf')

        spp_list = list(bath_cultures.groupby('Species').groups.keys())
        all_spp_comb_exps = list(combinations(spp_list, num_spp_exp))

        spp_comb_names = list(all_spp_comb_exps[exp_num])
        ini_r = bath_cultures.groupby('Species').first().NO3_uM.loc[spp_comb_names].values.max() * 1. / 1e6
        ini_d = list(bath_cultures.groupby('Species').first().Abun_cellmL.loc[spp_comb_names].values * 1e3 / 1.)
        spp_mean_size = allometries.groupby('Species').first().Vcell.loc[spp_comb_names].values
        ini_min_size = list(spp_mean_size - (spp_mean_size * rel_size_range))
        ini_max_size = list(spp_mean_size + (spp_mean_size * rel_size_range))
        spp_short_names = [sp[0] + sp[search('_', sp).span()[1]] for sp in spp_comb_names]
        arr_sp_short_names = np.array(spp_short_names)[np.newaxis, ...]
        arr_sp_long_names = np.array(spp_comb_names)[np.newaxis, ...]
        out_list = [ini_r, ini_d, ini_min_size, ini_max_size, spp_short_names, all_spp_comb_exps, arr_sp_short_names,
                    arr_sp_long_names]
        return out_list

    def create_dataset(exp_num, _short_names, _long_names, _sbmi=None, _sbmc=None):

        if _sbmi is not None and _sbmc is not None:
            ds = xr.Dataset(data_vars={
                'sbmi_agents_size': (
                    ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], _sbmi.agents_size[np.newaxis, ...]),
                'sbmi_agents_biomass': (
                    ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], _sbmi.agents_biomass[np.newaxis, ...]),
                'sbmi_agents_abundance': (
                    ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], _sbmi.agents_abundance[np.newaxis, ...]),
                'sbmi_agents_growth': (
                    ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], _sbmi.agents_growth[np.newaxis, ...]),
                'sbmi_resource': (['exp_num', 'time_sbmi'], _sbmi.resource[np.newaxis, ...]),
                'sbmi_tot_abundance': (['exp_num', 'time_sbmi'], _sbmi.abundance[np.newaxis, ...]),
                'sbmi_tot_biomass': (['exp_num', 'time_sbmi'], _sbmi.biomass[np.newaxis, ...]),
                'sbmi_tot_quota': (['exp_num', 'time_sbmi'], _sbmi.quota[np.newaxis, ...]),
                'sbmi_massbalance': (['exp_num', 'time_sbmi'], _sbmi.massbalance[np.newaxis, ...]),
                'sbmc_size': (['exp_num', 'idx_num_sc'], _sbmc._size_range[np.newaxis, ...]),
                'sbmc_biomass': (['exp_num', 'time_sbmc', 'idx_num_sc'], _sbmc.biomass[np.newaxis, ...]),
                'sbmc_abundance': (['exp_num', 'time_sbmc', 'idx_num_sc'], _sbmc.abundance[np.newaxis, ...]),
                'sbmc_quota': (['exp_num', 'time_sbmc', 'idx_num_sc'], _sbmc.quota[np.newaxis, ...]),
                'sbmc_growth': (['exp_num', 'time_sbmc', 'idx_num_sc'], _sbmc.mus[np.newaxis, ...]),
                'sbmc_resource': (['exp_num', 'time_sbmc'], _sbmc.resource[np.newaxis, ...])
            },
                coords={
                    'exp_num': [exp_num],
                    'spp_num': np.arange(num_spp_exp),
                    'spp_name_short': (['exp_num', 'spp_num'], _short_names),
                    'spp_name_long': (['exp_num', 'spp_num'], _long_names),
                    'time_sbmi': _sbmi.time,
                    'time_sbmc': _sbmc.time,
                    'num_agents': np.arange(sbmi_nsimax + (sbmi_nsimin * num_spp_exp)),
                    'idx_num_sc': np.arange(sum(sbmc_numsc * num_spp_exp))
                },
                attrs={'title': 'Multiple species experiments',
                       'description': 'Experiments for a combination of ' + str(num_spp_exp) +
                                      ' species and by using two size-based model types. '
                                      'One model type based on individuals (SBMi) '
                                      'and another type based on size classes (SBMc)',
                       'simulations setup': 'relative size range:' + str(rel_size_range) +
                                            ', dilution rate:' + str(dilution) +
                                            ', volume:' + str(volume) +
                                            ', maximum time of simulations:' + str(time_end) +
                                            ', initial number of size classes:' + str(sbmc_numsc) +
                                            ', initial number of (super) individuals:' + str(sbmi_nsispp) +
                                            ', minimum number of (super) individuals:' + str(sbmi_nsimin) +
                                            ', maximum number of (super) individuals:' + str(sbmi_nsimax) +
                                            ', time step for individual-based model simulations:' + str(sbmi_ts),
                       'time_units': 'd (days)',
                       'size_units': 'um^3 (cubic micrometers)',
                       'biomass_units': 'mol C / cell (mol of carbon per cell)',
                       'abundance_units': 'cell / L (cells per litre)',
                       'quota_units': 'mol N / mol C * cell (mol of nitrogen per mol of carbon per cell)',
                       'growth_units': '1 / day (per day)',
                       'resource_units': 'uM N (micro Molar of nitrogen)'
                       })
        elif _sbmi is not None and _sbmc is None:
            ds = xr.Dataset(data_vars={
                'sbmi_agents_size': (
                    ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], _sbmi.agents_size[np.newaxis, ...]),
                'sbmi_agents_biomass': (
                    ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], _sbmi.agents_biomass[np.newaxis, ...]),
                'sbmi_agents_abundance': (
                    ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], _sbmi.agents_abundance[np.newaxis, ...]),
                'sbmi_agents_growth': (
                    ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], _sbmi.agents_growth[np.newaxis, ...]),
                'sbmi_resource': (['exp_num', 'time_sbmi'], _sbmi.resource[np.newaxis, ...]),
                'sbmi_tot_abundance': (['exp_num', 'time_sbmi'], _sbmi.abundance[np.newaxis, ...]),
                'sbmi_tot_biomass': (['exp_num', 'time_sbmi'], _sbmi.biomass[np.newaxis, ...]),
                'sbmi_tot_quota': (['exp_num', 'time_sbmi'], _sbmi.quota[np.newaxis, ...]),
                'sbmi_massbalance': (['exp_num', 'time_sbmi'], _sbmi.massbalance[np.newaxis, ...]),
            },
                coords={
                    'exp_num': [exp_num],
                    'spp_num': np.arange(num_spp_exp),
                    'spp_name_short': (['exp_num', 'spp_num'], _short_names),
                    'spp_name_long': (['exp_num', 'spp_num'], _long_names),
                    'time_sbmi': _sbmi.time,
                    'num_agents': np.arange(sbmi_nsimax + (sbmi_nsimin * num_spp_exp))

                },
                attrs={'title': 'Multiple species experiments',
                       'description': 'Experiments for a combination of ' + str(num_spp_exp) +
                                      ' species and by using a size-based model types'
                                      ' based on individuals (SBMi) ',
                       'simulations setup': 'relative size range:' + str(rel_size_range) +
                                            ', dilution rate:' + str(dilution) +
                                            ', volume:' + str(volume) +
                                            ', maximum time of simulations:' + str(time_end) +
                                            ', initial number of (super) individuals:' + str(sbmi_nsispp) +
                                            ', minimum number of (super) individuals:' + str(sbmi_nsimin) +
                                            ', maximum number of (super) individuals:' + str(sbmi_nsimax) +
                                            ', time step for individual-based model simulations:' + str(sbmi_ts),
                       'time_units': 'd (days)',
                       'size_units': 'um^3 (cubic micrometers)',
                       'biomass_units': 'mol C / cell (mol of carbon per cell)',
                       'abundance_units': 'cell / L (cells per litre)',
                       'quota_units': 'mol N / mol C * cell (mol of nitrogen per mol of carbon per cell)',
                       'growth_units': '1 / day (per day)',
                       'resource_units': 'uM N (micro Molar of nitrogen)'
                       })
        elif _sbmc is not None and _sbmi is None:
            ds = xr.Dataset(data_vars={
                'sbmc_size': (['exp_num', 'idx_num_sc'], _sbmc._size_range[np.newaxis, ...]),
                'sbmc_biomass': (['exp_num', 'time_sbmc', 'idx_num_sc'], _sbmc.biomass[np.newaxis, ...]),
                'sbmc_abundance': (['exp_num', 'time_sbmc', 'idx_num_sc'], _sbmc.abundance[np.newaxis, ...]),
                'sbmc_quota': (['exp_num', 'time_sbmc', 'idx_num_sc'], _sbmc.quota[np.newaxis, ...]),
                'sbmc_growth': (['exp_num', 'time_sbmc', 'idx_num_sc'], _sbmc.mus[np.newaxis, ...]),
                'sbmc_resource': (['exp_num', 'time_sbmc'], _sbmc.resource[np.newaxis, ...])
            },
                coords={
                    'exp_num': [exp_num],
                    'spp_num': np.arange(num_spp_exp),
                    'spp_name_short': (['exp_num', 'spp_num'], _short_names),
                    'spp_name_long': (['exp_num', 'spp_num'], _long_names),
                    'time_sbmc': _sbmc.time,
                    'idx_num_sc': np.arange(sum(sbmc_numsc * num_spp_exp))
                },
                attrs={'title': 'Multiple species experiments',
                       'description': 'Experiments for a combination of ' + str(num_spp_exp) +
                                      ' species and by using two size-based model types '
                                      'based on size classes (SBMc)',
                       'simulations setup': 'relative size range:' + str(rel_size_range) +
                                            ', dilution rate:' + str(dilution) +
                                            ', volume:' + str(volume) +
                                            ', maximum time of simulations:' + str(time_end) +
                                            ', initial number of size classes:' + str(sbmc_numsc),
                       'time_units': 'd (days)',
                       'size_units': 'um^3 (cubic micrometers)',
                       'biomass_units': 'mol C / cell (mol of carbon per cell)',
                       'abundance_units': 'cell / L (cells per litre)',
                       'quota_units': 'mol N / mol C * cell (mol of nitrogen per mol of carbon per cell)',
                       'growth_units': '1 / day (per day)',
                       'resource_units': 'uM N (micro Molar of nitrogen)'
                       })
        return ds

    def save_ncfile(dataset, all_spp_list, exp_num):
        fpath = './Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp'
        Path(fpath).mkdir(parents=True, exist_ok=True)
        if sbmi_variant and sbmc_variant:
            path = fpath + '/Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp_' + str(exp_num).zfill(
                str(len(all_spp_list)).__len__()) + '.nc'
        if sbmi_variant:
            path = fpath + '/SBMi_multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp_' + str(exp_num).zfill(
                str(len(all_spp_list)).__len__()) + '.nc'
        if sbmc_variant:
            path = fpath + '/SBMc_multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp_' + str(exp_num).zfill(
                str(len(all_spp_list)).__len__()) + '.nc'
        return dataset.to_netcdf(path=path)

    ds_delayed = []
    for exp, comb in enumerate(combinations(np.arange(22), num_spp_exp)):
        init_params = dask.delayed(get_init_param)(exp)

        init_r = init_params[0]
        init_d = init_params[1]
        init_min_size = init_params[2]
        init_max_size = init_params[3]
        sp_short_names = init_params[4]
        all_spp_comb = init_params[5]
        out_sp_short_names = init_params[6]
        out_sp_long_names = init_params[7]

        if sbmc_variant:
            sbmc = dask.delayed(SBMc)(ini_resource=init_r,  # mol N/L
                                      ini_density=init_d,  # cells/L
                                      min_size=init_min_size,  # um^3
                                      max_size=init_max_size,  # um^3
                                      spp_names=sp_short_names,
                                      dilution_rate=dilution,
                                      volume=volume,
                                      num_sc=sbmc_numsc * num_spp_exp,
                                      time_end=time_end,
                                      timeit=False,
                                      vectorize=True
                                      )
        if sbmi_variant:
            sbmi = dask.delayed(SBMi)(ini_resource=init_r,  # mol N/L
                                      ini_density=init_d,  # cells/L
                                      min_size=init_min_size,  # um^3
                                      max_size=init_max_size,  # um^3
                                      spp_names=sp_short_names,
                                      dilution_rate=dilution,
                                      volume=volume,
                                      nsi_spp=sbmi_nsispp * num_spp_exp,
                                      nsi_min=sbmi_nsimin,
                                      nsi_max=sbmi_nsimax + (sbmi_nsimin * num_spp_exp),
                                      time_step=sbmi_ts,
                                      time_end=time_end,
                                      timeit=False
                                      )

        del init_d, init_r, init_min_size, init_max_size, sp_short_names

        if sbmi_variant and sbmc_variant:
            ds_exp = dask.delayed(create_dataset)(exp_num=exp, _short_names=out_sp_short_names,
                                                  _long_names=out_sp_long_names, _sbmi=sbmi, _sbmc=sbmc)
            del sbmi, sbmc
        if sbmc_variant and not sbmi_variant:
            ds_exp = dask.delayed(create_dataset)(exp_num=exp, _short_names=out_sp_short_names,
                                                  _long_names=out_sp_long_names, _sbmc=sbmc)
            del sbmc
        if sbmi_variant and not sbmc_variant:
            ds_exp = dask.delayed(create_dataset)(exp_num=exp, _short_names=out_sp_short_names,
                                                  _long_names=out_sp_long_names, _sbmi=sbmi)
            del sbmi
        del out_sp_short_names, out_sp_long_names

        ds_out = dask.delayed(save_ncfile)(dataset=ds_exp, all_spp_list=all_spp_comb, exp_num=exp)

        ds_delayed.append(ds_out)

        del ds_exp, ds_out, all_spp_comb

    # dask.compute(*ds_delayed)
    batches = list(partition_all(compute_batches, ds_delayed))
    for batch in batches:
        dask.compute(*batch)

    print('Total computation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))


def run_sims_futures(num_spp_exp, rel_size_range=0.25, dilution=0.0, volume=1.0, max_time=20,
                     sbmc_numsc=[10], sbmc_ts=100, sbmi_nsispp=[101], sbmi_nsimin=100, sbmi_nsimax=1000, sbmi_ts=1 / 24,
                     ssh=False, ssh_username=None, ssh_pw=None, nprocs=4, nthreads=1, mem_lim=2e9):
    """
    This is an experimental function similar to the above to calculate multiple
    species experiments but using dask.futures instead of dask.dealyed.
    """

    start_comp_time = time.time()
    start_datetime = datetime.now()
    print('Start of experiments for combinations of %.i species on' % num_spp_exp,
          start_datetime.strftime("%b-%d-%Y %H:%M:%S"))

    # try:
    #     client = Client('tcp://localhost:8786', timeout=1)
    # except OSError:
    if ssh:
        cluster = SSHCluster(["localhost", "localhost", "localhost", "localhost"],
                             connect_options={"known_hosts": None,
                                              "username": ssh_username,
                                              "password": ssh_pw},
                             worker_options={"nprocs": nprocs, "nthreads": nthreads, "memory_limit": mem_lim},
                             scheduler_options={"port": 0, "dashboard_address": ":8787"})
        client = Client(cluster)
    else:
        cluster = LocalCluster(n_workers=nprocs, threads_per_worker=nthreads, memory_limit=mem_lim,
                               scheduler_port=8786)
        client = Client(cluster)

    data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')

    def max_init_resource(cultures_dtf, spp_exp_names):
        return cultures_dtf.groupby('Species').first().NO3_uM.loc[spp_exp_names].values.max() * 1. / 1e6

    def spp_init_abundance(cultures_dtf, spp_exp_names):
        return list(cultures_dtf.groupby('Species').first().Abun_cellmL.loc[spp_exp_names].values * 1e3 / 1.)

    def spp_min_size(allometries_dtf, spp_exp_names):
        spp_mean_size = allometries_dtf.groupby('Species').first().Vcell.loc[spp_exp_names].values
        return list(spp_mean_size - (spp_mean_size * rel_size_range))

    def spp_max_size(allometries_dtf, spp_exp_names):
        spp_mean_size = allometries_dtf.groupby('Species').first().Vcell.loc[spp_exp_names].values
        return list(spp_mean_size + (spp_mean_size * rel_size_range))

    def short_names(spp_exp_names):
        return [sp[0] + sp[search('_', sp).span()[1]] for sp in spp_exp_names]

    def spp_all(cultures_dtf):
        return list(cultures_dtf.groupby('Species').groups.keys())

    def spp_comb(all_spp_list, exp_num):
        return list(all_spp_list[exp_num])

    def get_sbmc_size(sbmc_obj):
        return sbmc_obj._size_range[np.newaxis, ...]

    def get_sbmc_biomass(sbmc_obj):
        return sbmc_obj.biomass[np.newaxis, ...]

    def get_sbmc_abundance(sbmc_obj):
        return sbmc_obj.abundance[np.newaxis, ...]

    def get_sbmc_quota(sbmc_obj):
        return sbmc_obj.quota[np.newaxis, ...]

    def get_sbmc_growth(sbmc_obj):
        return sbmc_obj.mus[np.newaxis, ...]

    def get_sbmc_resource(sbmc_obj):
        return sbmc_obj.resource[np.newaxis, ...]

    def get_sbmi_agents_size(sbmi_obj):
        return sbmi_obj.agents_size[np.newaxis, ...]

    def get_sbmi_agents_biomass(sbmi_obj):
        return sbmi_obj.agents_biomass[np.newaxis, ...]

    def get_sbmi_agents_abundance(sbmi_obj):
        return sbmi_obj.agents_abundance[np.newaxis, ...]

    def get_sbmi_agents_growth(sbmi_obj):
        return sbmi_obj.agents_growth[np.newaxis, ...]

    def get_sbmi_resource(sbmi_obj):
        return sbmi_obj.resource[np.newaxis, ...]

    def get_sbmi_tot_abundance(sbmi_obj):
        return sbmi_obj.abundance[np.newaxis, ...]

    def get_sbmi_tot_biomass(sbmi_obj):
        return sbmi_obj.biomass[np.newaxis, ...]

    def get_sbmi_tot_quota(sbmi_obj):
        return sbmi_obj.quota[np.newaxis, ...]

    def get_sbmi_massbalance(sbmi_obj):
        return sbmi_obj.massbalance[np.newaxis, ...]

    def arr_sp_short_names(lst_sp_short_names):
        return np.array(lst_sp_short_names)[np.newaxis, ...]

    def arr_sp_long_names(lst_sp_long_names, exp_num):
        return np.array(lst_sp_long_names[exp_num])[np.newaxis, ...]

    def save_ncfile(dataset, all_spp_list, exp_num):
        fpath = './Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp'
        Path(fpath).mkdir(parents=True, exist_ok=True)
        path = fpath + '/Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp_' + str(exp_num).zfill(
            str(len(all_spp_list)).__len__()) + '.nc'
        return dataset.to_netcdf(path=path)

    for exp, comb in enumerate(combinations(np.arange(22), num_spp_exp)):
        allometries = client.scatter(pd.read_hdf(data_path, 'allodtf'))
        cultures = client.scatter(pd.read_hdf(data_path, 'batchdtf'))

        spp_list = client.submit(spp_all, cultures)
        all_spp_comb_iter = client.submit(combinations, spp_list, num_spp_exp)
        all_spp_comb = client.submit(list, all_spp_comb_iter)

        del spp_list, all_spp_comb_iter

        spp_exp_list = client.submit(spp_comb, all_spp_list=all_spp_comb, exp_num=exp)
        init_r = client.submit(max_init_resource, cultures_dtf=cultures, spp_exp_names=spp_exp_list)
        init_d = client.submit(spp_init_abundance, cultures_dtf=cultures, spp_exp_names=spp_exp_list)
        init_min_size = client.submit(spp_min_size, allometries_dtf=allometries, spp_exp_names=spp_exp_list)
        init_max_size = client.submit(spp_max_size, allometries_dtf=allometries, spp_exp_names=spp_exp_list)
        sp_short_names = client.submit(short_names, spp_exp_names=spp_exp_list)

        del spp_exp_list, allometries, cultures

        sbmc = client.submit(SBMc, ini_resource=init_r,  # mol N/L
                             ini_density=init_d,  # cells/L
                             min_size=init_min_size,  # um^3
                             max_size=init_max_size,  # um^3
                             spp_names=sp_short_names,
                             dilution_rate=dilution,
                             volume=volume,
                             num_sc=sbmc_numsc * num_spp_exp,
                             time_end=max_time,
                             timeit=False,
                             vectorize=True
                             )

        sbmi = client.submit(SBMi, ini_resource=init_r,  # mol N/L
                             ini_density=init_d,  # cells/L
                             min_size=init_min_size,  # um^3
                             max_size=init_max_size,  # um^3
                             spp_names=sp_short_names,
                             dilution_rate=dilution,
                             volume=volume,
                             nsi_spp=sbmi_nsispp * num_spp_exp,
                             nsi_min=sbmi_nsimin,
                             nsi_max=sbmi_nsimax + (sbmi_nsimin * num_spp_exp),
                             time_step=sbmi_ts,
                             time_end=max_time,
                             timeit=False
                             )

        del init_d, init_r, init_min_size, init_max_size

        sbmc_size = client.submit(get_sbmc_size, sbmc_obj=sbmc)
        sbmc_biomass = client.submit(get_sbmc_biomass, sbmc_obj=sbmc)
        sbmc_abundance = client.submit(get_sbmc_abundance, sbmc_obj=sbmc)
        sbmc_quota = client.submit(get_sbmc_quota, sbmc_obj=sbmc)
        sbmc_growth = client.submit(get_sbmc_growth, sbmc_obj=sbmc)
        sbmc_resource = client.submit(get_sbmc_resource, sbmc_obj=sbmc)

        sbmi_agents_size = client.submit(get_sbmi_agents_size, sbmi_obj=sbmi)
        sbmi_agents_biomass = client.submit(get_sbmi_agents_biomass, sbmi_obj=sbmi)
        sbmi_agents_abundance = client.submit(get_sbmi_agents_abundance, sbmi_obj=sbmi)
        sbmi_agents_growth = client.submit(get_sbmi_agents_growth, sbmi_obj=sbmi)
        sbmi_resource = client.submit(get_sbmi_resource, sbmi_obj=sbmi)
        sbmi_tot_abundance = client.submit(get_sbmi_tot_abundance, sbmi_obj=sbmi)
        sbmi_tot_biomass = client.submit(get_sbmi_tot_biomass, sbmi_obj=sbmi)
        sbmi_tot_quota = client.submit(get_sbmi_tot_quota, sbmi_obj=sbmi)
        sbmi_massbalance = client.submit(get_sbmi_massbalance, sbmi_obj=sbmi)

        out_sp_short_names = client.submit(arr_sp_short_names, lst_sp_short_names=sp_short_names)
        out_sp_long_names = client.submit(arr_sp_long_names, lst_sp_long_names=all_spp_comb, exp_num=exp)
        del sbmi, sbmc

        ds_exp = client.submit(xr.Dataset, data_vars={
            'sbmi_agents_size': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_size),
            'sbmi_agents_biomass': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_biomass),
            'sbmi_agents_abundance': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_abundance),
            'sbmi_agents_growth': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_growth),
            'sbmi_resource': (['exp_num', 'time_sbmi'], sbmi_resource),
            'sbmi_tot_abundance': (['exp_num', 'time_sbmi'], sbmi_tot_abundance),
            'sbmi_tot_biomass': (['exp_num', 'time_sbmi'], sbmi_tot_biomass),
            'sbmi_tot_quota': (['exp_num', 'time_sbmi'], sbmi_tot_quota),
            'sbmi_massbalance': (['exp_num', 'time_sbmi'], sbmi_massbalance),
            'sbmc_size': (['exp_num', 'idx_num_sc'], sbmc_size),
            'sbmc_biomass': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_biomass),
            'sbmc_abundance': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_abundance),
            'sbmc_quota': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_quota),
            'sbmc_growth': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_growth),
            'sbmc_resource': (['exp_num', 'time_sbmc'], sbmc_resource)
        }, coords={
            'exp_num': [exp],
            'spp_num': np.arange(num_spp_exp),
            'spp_name_short': (['exp_num', 'spp_num'], out_sp_short_names),
            'spp_name_long': (['exp_num', 'spp_num'], out_sp_long_names),
            'time_sbmi': np.arange(max_time + 1),
            'time_sbmc': np.linspace(0, max_time, sbmc_ts),
            'num_agents': np.arange(sbmi_nsimax + (sbmi_nsimin * num_spp_exp)),
            'idx_num_sc': np.arange(sum(sbmc_numsc * num_spp_exp))
        }, attrs={'title': 'Multiple species experiments',
                  'description': 'Experiments for a combination of ' + str(num_spp_exp) +
                                 ' species and by using two size-based model types. '
                                 'One model type based on individuals (SBMi) '
                                 'and another type based on size classes (SBMc)',
                  'simulations setup': 'relative size range:' + str(rel_size_range) +
                                       ', dilution rate:' + str(dilution) +
                                       ', volume:' + str(volume) +
                                       ', maximum time of simulations:' + str(max_time) +
                                       ', initial number of size classes:' + str(sbmc_numsc) +
                                       ', initial number of (super) individuals:' + str(sbmi_nsispp) +
                                       ', minimum number of (super) individuals:' + str(sbmi_nsimin) +
                                       ', maximum number of (super) individuals:' + str(sbmi_nsimax) +
                                       ', time step for individual-based model simulations:' + str(sbmi_ts),
                  'time_units': 'd (days)',
                  'size_units': 'um^3 (cubic micrometers)',
                  'biomass_units': 'mol C / cell (mol of carbon per cell)',
                  'abundance_units': 'cell / L (cells per litre)',
                  'quota_units': 'mol N / mol C * cell (mol of nitrogen per mol of carbon per cell)',
                  'growth_units': '1 / day (per day)',
                  'resource_units': 'uM N (micro Molar of nitrogen)'
                  })

        del sbmc_size, sbmc_biomass, sbmc_abundance, sbmc_quota, sbmc_growth, sbmc_resource
        del sbmi_agents_size, sbmi_agents_biomass, sbmi_agents_abundance, sbmi_agents_growth
        del sbmi_tot_biomass, sbmi_tot_quota, sbmi_tot_abundance, sbmi_resource, sbmi_massbalance
        del out_sp_short_names, out_sp_long_names

        ds_out = client.submit(save_ncfile, dataset=ds_exp, all_spp_list=all_spp_comb, exp_num=exp)

        del ds_exp, ds_out, all_spp_comb

    print('Total computation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))


def multiple_spp_sims(num_spp_exp, rel_size_range=0.25, dilution=0.0, volume=1.0, time_end=20,
                      sbmc_numsc=[10], sbmi_nsispp=[101], sbmi_nsimin=100, sbmi_nsimax=1000, sbmi_ts=1 / 24,
                      sbmc_variant=False, sbmi_asyn_variant=False, sbmi_syn_variant=False,
                      n_procs=6, n_threads=1, mem_lim=10e9):
    cluster = LocalCluster(n_workers=n_procs, threads_per_worker=n_threads, memory_limit=mem_lim)
    client = Client(cluster)
    print(client.dashboard_link)

    def get_init_param(exp_num):
        data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
        allometries = pd.read_hdf(data_path, 'allodtf')
        bath_cultures = pd.read_hdf(data_path, 'batchdtf')

        spp_list = list(bath_cultures.groupby('Species').groups.keys())
        all_spp_comb_exps = list(combinations(spp_list, num_spp_exp))

        spp_comb_names = list(all_spp_comb_exps[exp_num])
        ini_r = bath_cultures.groupby('Species').first().NO3_uM.loc[spp_comb_names].values.max() * 1. / 1e6
        ini_d = list(bath_cultures.groupby('Species').first().Abun_cellmL.loc[spp_comb_names].values * 1e3 / 1.)
        spp_mean_size = allometries.groupby('Species').first().Vcell.loc[spp_comb_names].values
        ini_min_size = list(spp_mean_size - (spp_mean_size * rel_size_range))
        ini_max_size = list(spp_mean_size + (spp_mean_size * rel_size_range))
        spp_short_names = [sp[0] + sp[search('_', sp).span()[1]] for sp in spp_comb_names]
        out_list = [ini_r, ini_d, ini_min_size, ini_max_size, spp_short_names]
        return out_list

    def save_to_file(sbmc, sbmi_asyn, sbmi_syn):
        if sbmc_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmc])
             .to_xarray()
             .to_zarr('Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp.zarr', group='sbmc')
             )
        if sbmi_asyn_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmi_asyn])
             .to_xarray()
             .to_zarr('Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp.zarr', group='sbmi_asyn')
             )
        if sbmi_syn_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmi_syn])
             .to_xarray()
             .to_zarr('Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp.zarr', group='sbmi_syn')
             )

    sbmc_out = []
    sbmi_asyn_out = []
    sbmi_syn_out = []
    for exp, comb in enumerate(combinations(np.arange(22), num_spp_exp)):

        init_params = dask.delayed(get_init_param)(exp)
        init_r = init_params[0]
        init_d = init_params[1]
        init_min_size = init_params[2]
        init_max_size = init_params[3]
        sp_short_names = init_params[4]

        if sbmc_variant:
            sbmc = dask.delayed(SBMc)(ini_resource=init_r,  # mol N/L
                                      ini_density=init_d,  # cells/L
                                      min_cell_size=init_min_size,  # um^3
                                      max_cell_size=init_max_size,  # um^3
                                      spp_names=sp_short_names,
                                      dilution_rate=dilution,
                                      volume=volume,
                                      num_sc=sbmc_numsc * num_spp_exp,
                                      time_end=time_end
                                      )
            sbmc_out.append(sbmc)
        if sbmi_asyn_variant:
            sbmi_asyn = dask.delayed(SBMi_asyn)(ini_resource=init_r,  # mol N/L
                                                ini_density=init_d,  # cells/L
                                                min_cell_size=init_min_size,  # um^3
                                                max_cell_size=init_max_size,  # um^3
                                                spp_names=sp_short_names,
                                                dilution_rate=dilution,
                                                volume=volume,
                                                nsi_spp=sbmi_nsispp * num_spp_exp,
                                                nsi_min=sbmi_nsimin,
                                                nsi_max=sbmi_nsimax + (sbmi_nsimin * num_spp_exp),
                                                time_step=sbmi_ts,
                                                time_end=time_end
                                                )
            sbmi_asyn_out.append(sbmi_asyn)
        if sbmi_syn_variant:
            sbmi_syn = dask.delayed(SBMi_syn)(ini_resource=init_r,  # mol N/L
                                              ini_density=init_d,  # cells/L
                                              min_cell_size=init_min_size,  # um^3
                                              max_cell_size=init_max_size,  # um^3
                                              spp_names=sp_short_names,
                                              dilution_rate=dilution,
                                              volume=volume,
                                              nsi_spp=sbmi_nsispp * num_spp_exp,
                                              nsi_min=sbmi_nsimin,
                                              nsi_max=sbmi_nsimax + (sbmi_nsimin * num_spp_exp),
                                              time_step=sbmi_ts,
                                              time_end=time_end
                                              )
            sbmi_syn_out.append(sbmi_syn)

    results = dask.delayed(save_to_file)(sbmc_out, sbmi_asyn_out, sbmi_syn_out)

    dask.compute(results)
    client.close()
    cluster.close()
