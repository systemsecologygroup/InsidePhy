from insidephy.size_based_models.SBMc import SBMc
from insidephy.size_based_models.SBMi import SBMi
import pandas as pd
import dask
from itertools import combinations
from re import search
import numpy as np
import xarray as xr
import time
from datetime import datetime
import pkg_resources
import dask.array as da
from pathlib import Path
from dask.distributed import Client, LocalCluster, SSHCluster


# dask.config.set(scheduler='processes')


def simulations(num_spp_exp, rel_size_range, dilution, volume, sbmc_numsc, max_time, sbmi_nsispp,
                sbmi_nsimin, sbmi_nsimax, sbmi_ts):
    data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    allometries = pd.read_hdf(data_path, 'allodtf')
    cultures = pd.read_hdf(data_path, 'batchdtf')

    spp_list = list(cultures.groupby('Species').groups.keys())
    spp_exps = list(combinations(spp_list, num_spp_exp))
    spp_resource = cultures.groupby('Species').first().NO3_uM
    spp_abundance = cultures.groupby('Species').first().Abun_cellmL * 1e3 / 1.
    spp_size = allometries.groupby('Species').first().Vcell

    init_r = [[spp_resource[sp] for sp in exp] for exp in spp_exps]
    init_d = [[spp_abundance[sp] for sp in exp] for exp in spp_exps]
    init_min_size = [[spp_size[sp] - (spp_size[sp] * rel_size_range) for sp in exp] for exp in spp_exps]
    init_max_size = [[spp_size[sp] + (spp_size[sp] * rel_size_range) for sp in exp] for exp in spp_exps]
    sp_short_names = [[sp[0] + sp[search('_', sp).span()[1]] for sp in exp] for exp in spp_exps]

    sbmc_out = []
    sbmi_out = []
    for c in range(len(spp_exps)):
        sbmc = dask.delayed(SBMc)(ini_resource=np.max(init_r[c]) * 1. / 1e6,  # mol N/L
                                  ini_density=init_d[c],  # cells/L
                                  min_size=init_min_size[c],  # um^3
                                  max_size=init_max_size[c],  # um^3
                                  spp_names=sp_short_names[c],
                                  dilution_rate=dilution,
                                  volume=volume,
                                  num_sc=sbmc_numsc,
                                  time_end=max_time,
                                  timeit=False,
                                  vectorize=True
                                  )
        sbmc_out.append(sbmc)
        sbmi = dask.delayed(SBMi)(ini_resource=np.max(init_r[c]) * 1. / 1e6,  # mol N/L
                                  ini_density=init_d[c],  # cells/L
                                  min_size=init_min_size[c],  # um^3
                                  max_size=init_max_size[c],  # um^3
                                  spp_names=sp_short_names[c],
                                  dilution_rate=dilution,
                                  volume=volume,
                                  nsi_spp=sbmi_nsispp,
                                  nsi_min=sbmi_nsimin,
                                  nsi_max=sbmi_nsimax,
                                  time_step=sbmi_ts,
                                  time_end=max_time,
                                  timeit=False
                                  )
        sbmi_out.append(sbmi)
    return dask.compute(sbmc_out, sbmi_out), sp_short_names


def run_exp_single_output(num_spp_exp, rel_size_range, dilution, volume, max_time,
                          sbmc_numsc, sbmi_nsispp, sbmi_nsimin, sbmi_nsimax, sbmi_ts):
    start_comp_time = time.time()
    start_datetime = datetime.now()
    out_filename = 'Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp' + '.nc'

    """return simulations(num_spp_exp=num_spp_exp, rel_size_range=rel_size_range,
                                                  dilution=dilution, volume=volume, maxtime=maxtime,
                                                  sbmc_numsc=sbmc_numsc, nsispp=sbmi_nsispp,
                                                  sbmi_nsimin=sbmi_nsimin, sbmi_nsimax=sbmi_nsimax, sbmi_ts=sbmi_ts)
    """
    (sbmc_exp, sbmi_exp), spp_names = simulations(num_spp_exp=num_spp_exp, rel_size_range=rel_size_range,
                                                  dilution=dilution, volume=volume, max_time=max_time,
                                                  sbmc_numsc=sbmc_numsc, sbmi_nsispp=sbmi_nsispp,
                                                  sbmi_nsimin=sbmi_nsimin, sbmi_nsimax=sbmi_nsimax, sbmi_ts=sbmi_ts)

    out_xr = xr.Dataset(data_vars={
        'sbmi_agents_size': (
            ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], da.from_array([exp.agents_size for exp in sbmi_exp])),
        'sbmi_agents_biomass': (
            ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], da.from_array([exp.agents_biomass for exp in sbmi_exp])),
        'sbmi_agents_abundance': (
            ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'],
            da.from_array([exp.agents_abundance for exp in sbmi_exp])),
        'sbmi_agents_growth': (
            ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], da.from_array([exp.agents_growth for exp in sbmi_exp])),
        'sbmi_resource': (['exp_num', 'time_sbmi'], da.from_array([exp.resource for exp in sbmi_exp])),
        'sbmi_tot_abundance': (['exp_num', 'time_sbmi'], da.from_array([exp.abundance for exp in sbmi_exp])),
        'sbmi_tot_biomass': (['exp_num', 'time_sbmi'], da.from_array([exp.biomass for exp in sbmi_exp])),
        'sbmi_tot_quota': (['exp_num', 'time_sbmi'], da.from_array([exp.quota for exp in sbmi_exp])),
        'sbmi_massbalance': (['exp_num', 'time_sbmi'], da.from_array([exp.massbalance for exp in sbmi_exp])),
        'sbmc_size': (['exp_num', 'idx_num_sc'], da.from_array([exp.size_range for exp in sbmc_exp])),
        'sbmc_biomass': (['exp_num', 'time_sbmc', 'idx_num_sc'], da.from_array([exp.biomass for exp in sbmc_exp])),
        'sbmc_abundance': (['exp_num', 'time_sbmc', 'idx_num_sc'], da.from_array([exp.abundance for exp in sbmc_exp])),
        'sbmc_quota': (['exp_num', 'time_sbmc', 'idx_num_sc'], da.from_array([exp.quota for exp in sbmc_exp])),
        'sbmc_growth': (['exp_num', 'time_sbmc', 'idx_num_sc'], da.from_array([exp.mus for exp in sbmc_exp])),
        'sbmc_resource': (['exp_num', 'time_sbmc'], da.from_array([exp.resource for exp in sbmc_exp])),

    }, coords={
        'exp_num': np.arange(len(sbmc_exp)),
        'exp_names': (['exp_num'], ['-'.join(spp_exp) for spp_exp in spp_names]),
        'num_spp_exp': np.arange(num_spp_exp),
        'spp_names': (['exp_num', 'num_spp_exp'], np.array(spp_names)),
        'time_sbmi': sbmi_exp[0].time,
        'time_sbmc': sbmc_exp[0].time,
        'num_agents': np.arange(sbmi_nsimax),
        'num_spp_sc': (['exp_num', 'num_spp_exp'], [exp.numsc for exp in sbmc_exp]),
    })
    out_xr.attrs['title'] = 'Multiple species experiments'
    out_xr.attrs['description'] = 'Experiments for a combination of ' + str(num_spp_exp) + \
                                  ' species and by using two size-based model types. ' \
                                  'One model type based on individuals (SBMi) ' \
                                  'and another type based on size classes (SBMc)'
    out_xr.attrs['simulations setup'] = 'relative size range:' + str(rel_size_range) + \
                                        ', dilution rate:' + str(dilution) + \
                                        ', volume:' + str(volume) + \
                                        ', maximum time of simulations:' + str(max_time) + \
                                        ', initial number of size classes:' + str(sbmc_numsc) + \
                                        ', initial number of (super) individuals:' + str(sbmi_nsispp) + \
                                        ', minimum number of (super) individuals:' + str(sbmi_nsimin) + \
                                        ', maximum number of (super) individuals:' + str(sbmi_nsimax) + \
                                        ', time step for individual-based model simulations:' + str(sbmi_ts)
    out_xr.attrs['time_units'] = 'd (days)'
    out_xr.attrs['size_units'] = 'um^3 (cubic micrometers)'
    out_xr.attrs['biomass_units'] = 'mol C / cell (mol of carbon per cell)'
    out_xr.attrs['abundance_units'] = 'cell / L (cells per litre)'
    out_xr.attrs['quota_units'] = 'mol N / mol C * cell (mol of nitrogen per mol of carbon per cell)'
    out_xr.attrs['biomass_units'] = 'mol C / cell (mol of carbon per cell)'
    out_xr.attrs['growth_units'] = '1 / day (per day)'
    out_xr.attrs['resource_units'] = 'uM N (micro Molar of nitrogen)'
    out_xr.attrs['start date'] = start_datetime.strftime("%b-%d-%Y %H:%M:%S")
    out_xr.attrs['total running time'] = '%.2f minutes' % ((time.time() - start_comp_time) / 60.)
    with dask.config.set(scheduler='threads'):
        out_xr.to_netcdf(out_filename)
    print('Completed experiments for combinations of %.i species simulations. '
          'Total computation time %.2f minutes' % (num_spp_exp, (time.time() - start_comp_time) / 60.))


def run_exp_multiple_output(num_spp_exp, rel_size_range, dilution, volume, sbmc_numsc, max_time, sbmi_nsispp,
                            sbmi_nsimin, sbmi_nsimax, sbmi_ts):
    start_comp_time = time.time()
    start_datetime = datetime.now()
    print('Start of experiments for combinations of %.i species simulations on' % num_spp_exp,
          start_datetime.strftime("%b-%d-%Y %H:%M:%S"))

    (sbmc_exp, sbmi_exp), spp_names = simulations(num_spp_exp=num_spp_exp, rel_size_range=rel_size_range,
                                                  dilution=dilution, volume=volume, max_time=max_time,
                                                  sbmc_numsc=sbmc_numsc, sbmi_nsispp=sbmi_nsispp,
                                                  sbmi_nsimin=sbmi_nsimin, sbmi_nsimax=sbmi_nsimax, sbmi_ts=sbmi_ts)

    files_path = './Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp'
    Path(files_path).mkdir(parents=True, exist_ok=True)

    for exp in range(len(sbmc_exp)):
        fname = 'Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp_' + str(exp).zfill(
            str(len(sbmc_exp)).__len__()) + '.nc'
        ds_exp = xr.Dataset(data_vars={
            'sbmi_agents_size': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_exp[exp].agents_size[np.newaxis, ...]),
            'sbmi_agents_biomass': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_exp[exp].agents_biomass[np.newaxis, ...]),
            'sbmi_agents_abundance': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_exp[exp].agents_abundance[np.newaxis, ...]),
            'sbmi_agents_growth': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_exp[exp].agents_growth[np.newaxis, ...]),
            'sbmi_resource': (['exp_num', 'time_sbmi'], sbmi_exp[exp].resource[np.newaxis, ...]),
            'sbmi_tot_abundance': (['exp_num', 'time_sbmi'], sbmi_exp[exp].abundance[np.newaxis, ...]),
            'sbmi_tot_biomass': (['exp_num', 'time_sbmi'], sbmi_exp[exp].biomass[np.newaxis, ...]),
            'sbmi_tot_quota': (['exp_num', 'time_sbmi'], sbmi_exp[exp].quota[np.newaxis, ...]),
            'sbmi_massbalance': (['exp_num', 'time_sbmi'], sbmi_exp[exp].massbalance[np.newaxis, ...]),
            'sbmc_size': (['exp_num', 'idx_num_sc'], sbmc_exp[exp].size_range[np.newaxis, ...]),
            'sbmc_biomass': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_exp[exp].biomass[np.newaxis, ...]),
            'sbmc_abundance': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_exp[exp].abundance[np.newaxis, ...]),
            'sbmc_quota': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_exp[exp].quota[np.newaxis, ...]),
            'sbmc_growth': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_exp[exp].mus[np.newaxis, ...]),
            'sbmc_resource': (['exp_num', 'time_sbmc'], sbmc_exp[exp].resource[np.newaxis, ...])
        }, coords={
            'exp_num': [exp],
            'spp_num': np.arange(num_spp_exp),
            'spp_name': (['exp_num', 'spp_num'], np.array(sbmi_exp[exp].spp_names)[np.newaxis, ...]),
            'time_sbmi': sbmi_exp[0].time,
            'time_sbmc': sbmc_exp[0].time,
            'num_agents': np.arange(sbmi_nsimax),
            'idx_num_sc': np.arange(sum(sbmc_numsc))
        })
        ds_exp.attrs['title'] = 'Multiple species experiments'
        ds_exp.attrs['description'] = 'Experiments for a combination of ' + str(num_spp_exp) + \
                                      ' species and by using two size-based model types. ' \
                                      'One model type based on individuals (SBMi) ' \
                                      'and another type based on size classes (SBMc)'
        ds_exp.attrs['simulations setup'] = 'relative size range:' + str(rel_size_range) + \
                                            ', dilution rate:' + str(dilution) + \
                                            ', volume:' + str(volume) + \
                                            ', maximum time of simulations:' + str(max_time) + \
                                            ', initial number of size classes:' + str(sbmc_numsc) + \
                                            ', initial number of (super) individuals:' + str(sbmi_nsispp) + \
                                            ', minimum number of (super) individuals:' + str(sbmi_nsimin) + \
                                            ', maximum number of (super) individuals:' + str(sbmi_nsimax) + \
                                            ', time step for individual-based model simulations:' + str(sbmi_ts)
        ds_exp.attrs['time_units'] = 'd (days)'
        ds_exp.attrs['size_units'] = 'um^3 (cubic micrometers)'
        ds_exp.attrs['biomass_units'] = 'mol C / cell (mol of carbon per cell)'
        ds_exp.attrs['abundance_units'] = 'cell / L (cells per litre)'
        ds_exp.attrs['quota_units'] = 'mol N / mol C * cell (mol of nitrogen per mol of carbon per cell)'
        ds_exp.attrs['biomass_units'] = 'mol C / cell (mol of carbon per cell)'
        ds_exp.attrs['growth_units'] = '1 / day (per day)'
        ds_exp.attrs['resource_units'] = 'uM N (micro Molar of nitrogen)'
        ds_exp.to_netcdf(files_path + '/' + fname)
    print('Total computation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))


def failed_futures1(num_spp_exp, rel_size_range=0.25, dilution=0.0, volume=1.0, max_time=20,
                    sbmc_numsc=[10], sbmc_ts=100, sbmi_nsispp=[101], sbmi_nsimin=100, sbmi_nsimax=1000, sbmi_ts=1 / 24,
                    ssh=False, ssh_username=None, ssh_pw=None, nprocs=4, nthreads=1, mem_lim=2e9):
    start_comp_time = time.time()
    start_datetime = datetime.now()
    print('Start of experiments for combinations of %.i species on' % num_spp_exp,
          start_datetime.strftime("%b-%d-%Y %H:%M:%S"))

    if ssh:
        cluster = SSHCluster(["localhost", "localhost", "localhost", "localhost"],
                             connect_options={"known_hosts": None,
                                              "username": ssh_username,
                                              "password": ssh_pw},
                             worker_options={"nprocs": nprocs, "nthreads": nthreads, "memory_limit": mem_lim},
                             scheduler_options={"port": 0, "dashboard_address": ":8787"})
        client = Client(cluster)
    else:
        cluster = LocalCluster(n_workers=nprocs, threads_per_worker=nthreads, memory_limit=mem_lim)
        client = Client(cluster)

    data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    allometries = pd.read_hdf(data_path, 'allodtf')
    cultures = pd.read_hdf(data_path, 'batchdtf')

    spp_list = list(cultures.groupby('Species').groups.keys())
    spp_resource = cultures.groupby('Species').first().NO3_uM
    spp_abundance = cultures.groupby('Species').first().Abun_cellmL * 1e3 / 1.
    spp_size = allometries.groupby('Species').first().Vcell

    del data_path, allometries, cultures

    spp_exps = list(combinations(spp_list, num_spp_exp))

    def max_init_r(spp_exp_names):
        return spp_resource.loc[spp_exp_names].values.max() * 1. / 1e6

    def short_names(spp_exp_names, all_exps=False):
        if all_exps:
            return [[sp[0] + sp[search('_', sp).span()[1]] for sp in exp] for exp in spp_exps]
        else:
            return [sp[0] + sp[search('_', sp).span()[1]] for sp in spp_exp_names]

    sbmc_out = []
    sbmi_out = []
    for c in range(len(spp_exps)):
        spp_exp_list = list(spp_exps[c])
        init_r = client.submit(max_init_r, spp_exp_list)
        init_d = client.submit(list, spp_abundance.loc[spp_exp_list].values)
        init_min_size = client.submit(list, spp_size.loc[spp_exp_list].values -
                                      (spp_size.loc[spp_exp_list].values * rel_size_range))
        init_max_size = client.submit(list, spp_size.loc[spp_exp_list].values +
                                      (spp_size.loc[spp_exp_list].values * rel_size_range))
        sp_short_names = client.submit(short_names, spp_exp_list)

        del spp_exp_list

        sbmc = client.submit(SBMc, ini_resource=init_r,  # mol N/L
                             ini_density=init_d,  # cells/L
                             min_size=init_min_size,  # um^3
                             max_size=init_max_size,  # um^3
                             spp_names=sp_short_names,
                             dilution_rate=dilution,
                             volume=volume,
                             num_sc=sbmc_numsc * num_spp_exp,
                             time_end=max_time,
                             time_steps=sbmc_ts,
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
        sbmc_out.append(sbmc)
        sbmi_out.append(sbmi)
        del init_d, init_r, init_min_size, init_max_size, sp_short_names, sbmc, sbmi

    sbmc_size_dat = client.scatter([future.result().size_range for future in sbmc_out])
    sbmc_size = client.submit(da.from_array, sbmc_size_dat)
    del sbmc_size_dat
    sbmc_biomass_dat = client.scatter([future.result().biomass for future in sbmc_out])
    sbmc_biomass = client.submit(da.from_array, sbmc_biomass_dat)
    del sbmc_biomass_dat
    sbmc_abundance_dat = client.scatter([future.result().abundance for future in sbmc_out])
    sbmc_abundance = client.submit(da.from_array, sbmc_abundance_dat)
    del sbmc_abundance_dat
    sbmc_quota_dat = client.scatter([future.result().quota for future in sbmc_out])
    sbmc_quota = client.submit(da.from_array, sbmc_quota_dat)
    del sbmc_quota_dat
    sbmc_growth_dat = client.scatter([future.result().mus for future in sbmc_out])
    sbmc_growth = client.submit(da.from_array, sbmc_growth_dat)
    del sbmc_growth_dat
    sbmc_resoruce_dat = client.scatter([future.result().resource for future in sbmc_out])
    sbmc_resoruce = client.submit(da.from_array, sbmc_resoruce_dat)
    del sbmc_resoruce_dat

    del sbmc_out

    sbmi_agents_size_dat = client.scatter([future.result().agents_size for future in sbmi_out])
    sbmi_agents_size = client.submit(da.from_array, sbmi_agents_size_dat)
    del sbmi_agents_size_dat
    sbmi_agents_biomass_dat = client.scatter([future.result().agents_biomass for future in sbmi_out])
    sbmi_agents_biomass = client.submit(da.from_array, sbmi_agents_biomass_dat)
    del sbmi_agents_biomass_dat
    sbmi_agents_abundance_dat = client.scatter([future.result().agents_abundance for future in sbmi_out])
    sbmi_agents_abundance = client.submit(da.from_array, sbmi_agents_abundance_dat)
    del sbmi_agents_abundance_dat
    sbmi_agents_growth_dat = client.scatter([future.result().agents_growth for future in sbmi_out])
    sbmi_agents_growth = client.submit(da.from_array, sbmi_agents_growth_dat)
    del sbmi_agents_growth_dat
    sbmi_resource_dat = client.scatter([future.result().resource for future in sbmi_out])
    sbmi_resource = client.submit(da.from_array, sbmi_resource_dat)
    del sbmi_resource_dat
    sbmi_tot_abundance_dat = client.scatter([future.result().abundance for future in sbmi_out])
    sbmi_tot_abundance = client.submit(da.from_array, sbmi_tot_abundance_dat)
    del sbmi_tot_abundance_dat
    sbmi_tot_biomass_dat = client.scatter([future.result().biomass for future in sbmi_out])
    sbmi_tot_biomass = client.submit(da.from_array, sbmi_tot_biomass_dat)
    del sbmi_tot_biomass_dat
    sbmi_tot_quota_dat = client.scatter([future.result().quota for future in sbmi_out])
    sbmi_tot_quota = client.submit(da.from_array, sbmi_tot_quota_dat)
    del sbmi_tot_quota_dat
    sbmi_massbalance_dat = client.scatter([future.result().massbalance for future in sbmi_out])
    sbmi_massbalance = client.submit(da.from_array, sbmi_massbalance_dat)
    del sbmi_massbalance_dat

    del sbmi_out

    ds_sims = xr.Dataset(data_vars={
        'sbmi_agents_size': (['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_size.result()),
        'sbmi_agents_biomass': (['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_biomass.result()),
        'sbmi_agents_abundance': (['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_abundance.result()),
        'sbmi_agents_growth': (['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_growth.result()),
        'sbmi_resource': (['exp_num', 'time_sbmi'], sbmi_resource.result()),
        'sbmi_tot_abundance': (['exp_num', 'time_sbmi'], sbmi_tot_abundance.result()),
        'sbmi_tot_biomass': (['exp_num', 'time_sbmi'], sbmi_tot_biomass.result()),
        'sbmi_tot_quota': (['exp_num', 'time_sbmi'], sbmi_tot_quota.result()),
        'sbmi_massbalance': (['exp_num', 'time_sbmi'], sbmi_massbalance.result()),
        'sbmc_size': (['exp_num', 'idx_num_sc'], sbmc_size.result()),
        'sbmc_biomass': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_biomass.result()),
        'sbmc_abundance': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_abundance.result()),
        'sbmc_quota': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_quota.result()),
        'sbmc_growth': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_growth.result()),
        'sbmc_resource': (['exp_num', 'time_sbmc'], sbmc_resoruce.result())
    },
        coords={
            'exp_num': np.arange(len(spp_exps)),
            'spp_num': np.arange(num_spp_exp),
            'spp_name_short': (['exp_num', 'spp_num'], np.array(short_names(spp_exps, all_exps=True))),
            'spp_name_long': (['exp_num', 'spp_num'], np.array(spp_exps)),
            'time_sbmi': np.arange(max_time + 1),
            'time_sbmc': np.linspace(0, max_time, sbmc_ts),
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
                                    ', maximum time of simulations:' + str(max_time) +
                                    ', initial number of size classes:' + str(sbmc_numsc) +
                                    ', initial number of (super) individuals:' + str(sbmi_nsispp) +
                                    ', minimum number of (super) individuals:' + str(sbmi_nsimin) +
                                    ', maximum number of (super) individuals:' + str(sbmi_nsimax) +
                                    ', time step for individual-based model:' + str(sbmi_ts),
               'time_units': 'd (days)',
               'size_units': 'um^3 (cubic micrometers)',
               'biomass_units': 'mol C / cell (mol of carbon per cell)',
               'abundance_units': 'cell / L (cells per litre)',
               'quota_units': 'mol N / mol C * cell (mol of nitrogen per mol of carbon per cell)',
               'growth_units': '1 / day (per day)',
               'resource_units': 'uM N (micro Molar of nitrogen)'
               })

    fpath = './Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp'
    Path(fpath).mkdir(parents=True, exist_ok=True)
    exps, datasets = zip(*ds_sims.groupby("exp_num"))
    paths = [fpath + '/Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp_' + str(e).zfill(
        str(len(spp_exps)).__len__()) + '.nc' for e in exps]
    xr.save_mfdataset(datasets, paths)
    print('Total computation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))


def failed_futures2(num_spp_exp, rel_size_range=0.25, dilution=0.0, volume=1.0, max_time=20,
                    sbmc_numsc=[10], sbmc_ts=100, sbmi_nsispp=[101], sbmi_nsimin=100, sbmi_nsimax=1000, sbmi_ts=1 / 24,
                    ssh=False, ssh_username=None, ssh_pw=None, nprocs=4, nthreads=1, mem_lim=2e9):
    start_comp_time = time.time()
    start_datetime = datetime.now()
    print('Start of experiments for combinations of %.i species on' % num_spp_exp,
          start_datetime.strftime("%b-%d-%Y %H:%M:%S"))

    if ssh:
        cluster = SSHCluster(["localhost", "localhost", "localhost", "localhost"],
                             connect_options={"known_hosts": None,
                                              "username": ssh_username,
                                              "password": ssh_pw},
                             worker_options={"nprocs": nprocs, "nthreads": nthreads, "memory_limit": mem_lim},
                             scheduler_options={"port": 0, "dashboard_address": ":8787"})
        client = Client(cluster)
    else:
        cluster = LocalCluster(n_workers=nprocs, threads_per_worker=nthreads, memory_limit=mem_lim)
        client = Client(cluster)

    data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    allometries = pd.read_hdf(data_path, 'allodtf')
    cultures = pd.read_hdf(data_path, 'batchdtf')

    spp_list = list(cultures.groupby('Species').groups.keys())
    spp_resource = cultures.groupby('Species').first().NO3_uM
    spp_abundance = cultures.groupby('Species').first().Abun_cellmL * 1e3 / 1.
    spp_size = allometries.groupby('Species').first().Vcell

    del data_path, allometries, cultures

    spp_exps = list(combinations(spp_list, num_spp_exp))

    def max_init_r(spp_exp_names):
        return spp_resource.loc[spp_exp_names].values.max() * 1. / 1e6

    def short_names(spp_exp_names, all_exps=False):
        if all_exps:
            return [[sp[0] + sp[search('_', sp).span()[1]] for sp in exp] for exp in spp_exps]
        else:
            return [sp[0] + sp[search('_', sp).span()[1]] for sp in spp_exp_names]

    sbmc_out = []
    sbmi_out = []
    for c in range(len(spp_exps)):
        spp_exp_list = list(spp_exps[c])
        init_r = client.submit(max_init_r, spp_exp_list)
        init_d = client.submit(list, spp_abundance.loc[spp_exp_list].values)
        init_min_size = client.submit(list, spp_size.loc[spp_exp_list].values -
                                      (spp_size.loc[spp_exp_list].values * rel_size_range))
        init_max_size = client.submit(list, spp_size.loc[spp_exp_list].values +
                                      (spp_size.loc[spp_exp_list].values * rel_size_range))
        sp_short_names = client.submit(short_names, spp_exp_list)

        del spp_exp_list

        sbmc = client.submit(SBMc, ini_resource=init_r,  # mol N/L
                             ini_density=init_d,  # cells/L
                             min_size=init_min_size,  # um^3
                             max_size=init_max_size,  # um^3
                             spp_names=sp_short_names,
                             dilution_rate=dilution,
                             volume=volume,
                             num_sc=sbmc_numsc * num_spp_exp,
                             time_end=max_time,
                             time_steps=sbmc_ts,
                             timeit=False,
                             vectorize=True
                             )

        sbmc_size = client.submit(da.from_array, sbmc.result().size_range)
        sbmc_biomass = client.submit(da.from_array, sbmc.result().biomass)
        sbmc_abundance = client.submit(da.from_array, sbmc.result().abundance)
        sbmc_quota = client.submit(da.from_array, sbmc.result().quota)
        sbmc_growth = client.submit(da.from_array, sbmc.result().mus)
        sbmc_resource = client.submit(da.from_array, sbmc.result().resource)

        del sbmc
        sbmc_out.append([sbmc_size, sbmc_biomass, sbmc_abundance, sbmc_quota, sbmc_growth, sbmc_resource])

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

        del init_d, init_r, init_min_size, init_max_size, sp_short_names

        sbmi_agents_size = client.submit(da.from_array, sbmi.result().agents_size)
        sbmi_agents_biomass = client.submit(da.from_array, sbmi.result().agents_biomass)
        sbmi_agents_abundance = client.submit(da.from_array, sbmi.result().agents_abundance)
        sbmi_agents_growth = client.submit(da.from_array, sbmi.result().agents_growth)
        sbmi_resource = client.submit(da.from_array, sbmi.result().resource)
        sbmi_tot_abundance = client.submit(da.from_array, sbmi.result().abundance)
        sbmi_tot_biomass = client.submit(da.from_array, sbmi.result().biomass)
        sbmi_tot_quota = client.submit(da.from_array, sbmi.result().quota)
        sbmi_massbalance = client.submit(da.from_array, sbmi.result().massbalance)

        del sbmi
        sbmi_out.append([sbmi_agents_size, sbmi_agents_biomass, sbmi_agents_abundance, sbmi_agents_growth,
                         sbmi_resource, sbmi_tot_abundance, sbmi_tot_biomass, sbmi_tot_quota, sbmi_massbalance])

    sbmc_size_dat = client.scatter([exp[0].result() for exp in sbmc_out])
    sbmc_size = client.submit(da.from_array, sbmc_size_dat)
    del sbmc_size_dat
    sbmc_biomass_dat = client.scatter([exp[1].result() for exp in sbmc_out])
    sbmc_biomass = client.submit(da.from_array, sbmc_biomass_dat)
    del sbmc_biomass_dat
    sbmc_abundance_dat = client.scatter([exp[2].result() for exp in sbmc_out])
    sbmc_abundance = client.submit(da.from_array, sbmc_abundance_dat)
    del sbmc_abundance_dat
    sbmc_quota_dat = client.scatter([exp[3].result() for exp in sbmc_out])
    sbmc_quota = client.submit(da.from_array, sbmc_quota_dat)
    del sbmc_quota_dat
    sbmc_growth_dat = client.scatter([exp[4].result() for exp in sbmc_out])
    sbmc_growth = client.submit(da.from_array, sbmc_growth_dat)
    del sbmc_growth_dat
    sbmc_resource_dat = client.scatter([exp[5].result() for exp in sbmc_out])
    sbmc_resource = client.submit(da.from_array, sbmc_resource_dat)
    del sbmc_resource_dat

    del sbmc_out

    sbmi_agents_size_dat = client.scatter([exp[0].result() for exp in sbmi_out])
    sbmi_agents_size = client.submit(da.from_array, sbmi_agents_size_dat)
    del sbmi_agents_size_dat
    sbmi_agents_biomass_dat = client.scatter([exp[1].result() for exp in sbmi_out])
    sbmi_agents_biomass = client.submit(da.from_array, sbmi_agents_biomass_dat)
    del sbmi_agents_biomass_dat
    sbmi_agents_abundance_dat = client.scatter([exp[2].result() for exp in sbmi_out])
    sbmi_agents_abundance = client.submit(da.from_array, sbmi_agents_abundance_dat)
    del sbmi_agents_abundance_dat
    sbmi_agents_growth_dat = client.scatter([exp[3].result() for exp in sbmi_out])
    sbmi_agents_growth = client.submit(da.from_array, sbmi_agents_growth_dat)
    del sbmi_agents_growth_dat
    sbmi_resource_dat = client.scatter([exp[4].result() for exp in sbmi_out])
    sbmi_resource = client.submit(da.from_array, sbmi_resource_dat)
    del sbmi_resource_dat
    sbmi_tot_abundance_dat = client.scatter([exp[5].result() for exp in sbmi_out])
    sbmi_tot_abundance = client.submit(da.from_array, sbmi_tot_abundance_dat)
    del sbmi_tot_abundance_dat
    sbmi_tot_biomass_dat = client.scatter([exp[6].result() for exp in sbmi_out])
    sbmi_tot_biomass = client.submit(da.from_array, sbmi_tot_biomass_dat)
    del sbmi_tot_biomass_dat
    sbmi_tot_quota_dat = client.scatter([exp[7].result() for exp in sbmi_out])
    sbmi_tot_quota = client.submit(da.from_array, sbmi_tot_quota_dat)
    del sbmi_tot_quota_dat
    sbmi_massbalance_dat = client.scatter([exp[8].result() for exp in sbmi_out])
    sbmi_massbalance = client.submit(da.from_array, sbmi_massbalance_dat)
    del sbmi_massbalance_dat

    del sbmi_out

    ds_sims = xr.Dataset(data_vars={
        'sbmi_agents_size': (['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_size.result()),
        'sbmi_agents_biomass': (['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_biomass.result()),
        'sbmi_agents_abundance': (['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_abundance.result()),
        'sbmi_agents_growth': (['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi_agents_growth.result()),
        'sbmi_resource': (['exp_num', 'time_sbmi'], sbmi_resource.result()),
        'sbmi_tot_abundance': (['exp_num', 'time_sbmi'], sbmi_tot_abundance.result()),
        'sbmi_tot_biomass': (['exp_num', 'time_sbmi'], sbmi_tot_biomass.result()),
        'sbmi_tot_quota': (['exp_num', 'time_sbmi'], sbmi_tot_quota.result()),
        'sbmi_massbalance': (['exp_num', 'time_sbmi'], sbmi_massbalance.result()),
        'sbmc_size': (['exp_num', 'idx_num_sc'], sbmc_size.result()),
        'sbmc_biomass': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_biomass.result()),
        'sbmc_abundance': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_abundance.result()),
        'sbmc_quota': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_quota.result()),
        'sbmc_growth': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc_growth.result()),
        'sbmc_resource': (['exp_num', 'time_sbmc'], sbmc_resource.result())
    }, coords={
        'exp_num': np.arange(len(spp_exps)),
        'spp_num': np.arange(num_spp_exp),
        'spp_name_short': (['exp_num', 'spp_num'], np.array(short_names(spp_exps, all_exps=True))),
        'spp_name_long': (['exp_num', 'spp_num'], np.array(spp_exps)),
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
                                   ', time step for individual-based model:' + str(sbmi_ts),
              'time_units': 'd (days)',
              'size_units': 'um^3 (cubic micrometers)',
              'biomass_units': 'mol C / cell (mol of carbon per cell)',
              'abundance_units': 'cell / L (cells per litre)',
              'quota_units': 'mol N / mol C * cell (mol of nitrogen per mol of carbon per cell)',
              'growth_units': '1 / day (per day)',
              'resource_units': 'uM N (micro Molar of nitrogen)'
              })

    fpath = './Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp'
    Path(fpath).mkdir(parents=True, exist_ok=True)
    exps, datasets = zip(*ds_sims.groupby("exp_num"))
    paths = [fpath + '/Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp_' + str(e).zfill(
        str(len(spp_exps)).__len__()) + '.nc' for e in exps]
    xr.save_mfdataset(datasets, paths)
    print('Total computation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))


def run_sims_delayed(num_spp_exp, rel_size_range=0.25, dilution=0.0, volume=1.0, max_time=20,
                     sbmc_numsc=[10], sbmi_nsispp=[101], sbmi_nsimin=100, sbmi_nsimax=1000, sbmi_ts=1 / 24,
                     ssh=False, ssh_username=None, ssh_pw=None, nprocs=4, nthreads=1, mem_lim=2e9):
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
        out_list = [ini_r, ini_d, ini_min_size, ini_max_size, spp_short_names, all_spp_comb_exps, arr_sp_short_names, arr_sp_long_names]
        return out_list

    def save_ncfile(dataset, all_spp_list, exp_num):
        fpath = './Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp'
        Path(fpath).mkdir(parents=True, exist_ok=True)
        path = fpath + '/Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp_' + str(exp_num).zfill(
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

        sbmc = dask.delayed(SBMc)(ini_resource=init_r,  # mol N/L
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
                                  time_end=max_time,
                                  timeit=False
                                  )

        del init_d, init_r, init_min_size, init_max_size

        ds_exp = dask.delayed(xr.Dataset)(data_vars={
            'sbmi_agents_size': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi.agents_size[np.newaxis, ...]),
            'sbmi_agents_biomass': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi.agents_biomass[np.newaxis, ...]),
            'sbmi_agents_abundance': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi.agents_abundance[np.newaxis, ...]),
            'sbmi_agents_growth': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], sbmi.agents_growth[np.newaxis, ...]),
            'sbmi_resource': (['exp_num', 'time_sbmi'], sbmi.resource[np.newaxis, ...]),
            'sbmi_tot_abundance': (['exp_num', 'time_sbmi'], sbmi.abundance[np.newaxis, ...]),
            'sbmi_tot_biomass': (['exp_num', 'time_sbmi'], sbmi.biomass[np.newaxis, ...]),
            'sbmi_tot_quota': (['exp_num', 'time_sbmi'], sbmi.quota[np.newaxis, ...]),
            'sbmi_massbalance': (['exp_num', 'time_sbmi'], sbmi.massbalance[np.newaxis, ...]),
            'sbmc_size': (['exp_num', 'idx_num_sc'], sbmc.size_range[np.newaxis, ...]),
            'sbmc_biomass': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc.biomass[np.newaxis, ...]),
            'sbmc_abundance': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc.abundance[np.newaxis, ...]),
            'sbmc_quota': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc.quota[np.newaxis, ...]),
            'sbmc_growth': (['exp_num', 'time_sbmc', 'idx_num_sc'], sbmc.mus[np.newaxis, ...]),
            'sbmc_resource': (['exp_num', 'time_sbmc'], sbmc.resource[np.newaxis, ...])
        }, coords={
            'exp_num': [exp],
            'spp_num': np.arange(num_spp_exp),
            'spp_name_short': (['exp_num', 'spp_num'], out_sp_short_names),
            'spp_name_long': (['exp_num', 'spp_num'], out_sp_long_names),
            'time_sbmi': sbmi.time,
            'time_sbmc': sbmc.time,
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

        del sbmi, sbmc, out_sp_short_names, out_sp_long_names

        ds_out = dask.delayed(save_ncfile)(ds_exp, all_spp_comb, exp)

        ds_delayed.append(ds_out)

        del ds_exp, ds_out, all_spp_comb, sp_short_names

    dask.compute(*ds_delayed)
    print('Total computation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))
    # client.close()


def run_sims_futures(num_spp_exp, rel_size_range=0.25, dilution=0.0, volume=1.0, max_time=20,
                     sbmc_numsc=[10], sbmc_ts=100, sbmi_nsispp=[101], sbmi_nsimin=100, sbmi_nsimax=1000, sbmi_ts=1 / 24,
                     ssh=False, ssh_username=None, ssh_pw=None, nprocs=4, nthreads=1, mem_lim=2e9):
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
        return sbmc_obj.size_range[np.newaxis, ...]

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
    # client.close()


"""
from insidephy.examples.multiple_spp_exp import run_sims_delayed, run_sims_futures

run_sims_delayed(num_spp_exp=2, max_time=10, sbmi_nsispp=[11],
        sbmi_nsimin=10, sbmi_nsimax=100, sbmi_ts=1/2)

run_sims_futures(num_spp_exp=2, max_time=10, sbmi_nsispp=[11],
        sbmi_nsimin=10, sbmi_nsimax=100, sbmi_ts=1/2)


"""

"""
from dask.distributed import Client, LocalCluster
import pkg_resources
from itertools import combinations
import dask.array as da
import pandas as pd
import dask
from re import search

test=sims(6)

futures = client.compute(test)



"""

"""
(sbmc_exp, sbmi_exp), spp_names = simulations(num_spp_exp=2, rel_size_range=0.25,
                                                  dilution=0.0, volume=1.0, max_time=10,
                                                  sbmc_numsc=2*[10], sbmi_nsispp=2*[11],
                                                  sbmi_nsimin=10, sbmi_nsimax=100+10*2, sbmi_ts=1/2)
"""

"""
run_exp2(num_spp_exp=2, rel_size_range=0.25,
           dilution=0.0, volume=1.0, max_time=10,
           sbmc_numsc=2*[10], sbmi_nsispp=2*[11],
           sbmi_nsimin=10, sbmi_nsimax=100+10*2, sbmi_ts=1/2)
"""

"""
ds0 = xr.open_dataset('./Multiple_spp_exp_02spp/Multiple_spp_exp_02spp_000.nc')

import xarray as xr
all_ds = xr.open_mfdataset('./Multiple_spp_exp_02spp/*.nc', combine='by_coords', concat_dim='exp_num')
"""

"""
from insidephy.examples.multiple_spp_exp import run_sims
from dask.distributed import Client, LocalCluster
import dask

cluster = LocalCluster(n_workers=4, threads_per_worker=1)
client = Client(cluster)
client.dashboard_link

dask.config.set(scheduler='process')
datasets=run_sims(num_spp_exp=2, rel_size_range=0.25,
        dilution=0.0, volume=1.0, max_time=10,
        sbmc_numsc=2*[10], sbmi_nsispp=2*[11],
        sbmi_nsimin=10, sbmi_nsimax=100+10*2, sbmi_ts=1/2)
        
datasets=client.compute(datasets)
fpath = './Multiple_spp_exp_' + str(2).zfill(2) + 'spp'   
files_paths = [fpath + '/Multiple_spp_exp_' + str(2).zfill(2) + 'spp_' + str(c).zfill(
str(231).__len__()) + '.nc' for c in range(231)]
xr.save_mfdataset(datasets, files_paths)

"""
