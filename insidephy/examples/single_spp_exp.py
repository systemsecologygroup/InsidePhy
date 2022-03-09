# from insidephy.size_based_models.SBMc import SBMc
# from insidephy.size_based_models.SBMi import SBMi
from insidephy.size_based_models.sbm import SBMc, SBMi_asyn, SBMi_syn
import pandas as pd
import dask
from re import search
import numpy as np
import xarray as xr
import time
from datetime import datetime
import pkg_resources
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster


def single_spp_sims(rel_size_range, dilution=0.0, volume=1, time_end=20,
                    sbmc_numsc=[1], sbmi_nsispp=[500], sbmi_nsimin=100,
                    sbmi_nsimax=1000, sbmi_ts=1 / (24 * 60), init_size=('mean', 'min'),
                    n_procs=6, n_threads=1, mem_lim=10e9,
                    sbmc_variant=False, sbmi_asyn_variant=False, sbmi_syn_variant=False):
    """
    function to calculate single species experiments from 22 phytoplankton species data
    reported in Marañón et al. (2013 Eco. Lett.).
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
    :param init_size: string
        whether to use minimum (min) or mean (mean) cell size to initialize species
        as reported by Marañón et al. (2013 Eco. Lett.)
    :param n_procs: integer
        number of cpus to compute the simulations
    :param n_threads:
        number of threads per cpu to use in the computations
    :param mem_lim:
        maximum memory usage per cpu
    :param sbmi_syn_variant: bool
        whether to calculate a model based on individual SBMi synchronous update
    :param sbmi_asyn_variant: bool
        whether to calculate a model based on individual SBMi asynchronous update
    :param sbmc_variant: bool
        whether to calculate a model based on size classes SBMc
    :return: zarr file
        single species simulation results
    """

    def save_to_file(sbmc, sbmi_asyn, sbmi_syn):
        if sbmc_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmc])
             .to_xarray()
             .to_zarr('Single_spp_exp_' + str(rel_size_range).zfill(2) + '.zarr', group='sbmc')
             )
        if sbmi_asyn_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmi_asyn])
             .to_xarray()
             .to_zarr('Single_spp_exp_' + str(rel_size_range).zfill(2) + '.zarr', group='sbmi_asyn')
             )
        if sbmi_syn_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmi_syn])
             .to_xarray()
             .to_zarr('Single_spp_exp_' + str(rel_size_range).zfill(2) + '.zarr', group='sbmi_syn')
             )

    cluster = LocalCluster(n_workers=n_procs, threads_per_worker=n_threads, memory_limit=mem_lim)
    client = Client(cluster)
    print(client.dashboard_link)

    data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    allometries = pd.read_hdf(data_path, 'allodtf')
    cultures = pd.read_hdf(data_path, 'batchdtf')
    cellsize = pd.read_hdf(data_path, 'sizedtf')

    spp_list = cultures.Species.cat.categories.to_list()
    spp_resource = cultures.groupby('Species').first().NO3_uM
    spp_abundance = cultures.groupby('Species').first().Abun_cellmL * 1e3 / 1.
    if init_size == 'mean':
        spp_size = allometries.groupby('Species').first().Vcell
    elif init_size == 'min':
        spp_size = cellsize.groupby('Species').min().Vcell
    elif isinstance(init_size, tuple):
        raise TypeError('Error on input parameter init_size, please specify the initial size'
                        ' as mean or min.')
    else:
        raise TypeError('Error on input parameter init_size, please specify the initial size'
                        ' as mean or min.')

    init_min_size = [spp_size[sp] - (spp_size[sp] * rel_size_range) for sp in spp_list]
    init_max_size = [spp_size[sp] + (spp_size[sp] * rel_size_range) for sp in spp_list]
    sp_short_names = [sp[0] + sp[search('_', sp).span()[1]] for sp in spp_list]
    sbmc_out = []
    sbmi_asyn_out = []
    sbmi_syn_out = []
    for sp in range(len(spp_list)):
        if sbmc_variant:
            sbmc = dask.delayed(SBMc)(ini_resource=spp_resource[sp] * 1. / 1e6,  # mol N/L
                                      ini_density=[spp_abundance[sp]],  # cells/L
                                      min_cell_size=[init_min_size[sp]],  # um^3
                                      max_cell_size=[init_max_size[sp]],  # um^3
                                      spp_names=[sp_short_names[sp]],
                                      dilution_rate=dilution,
                                      volume=volume,
                                      num_sc=sbmc_numsc,
                                      time_end=time_end
                                      )
            sbmc_out.append(sbmc)
        if sbmi_asyn_variant:
            sbmi_asyn = dask.delayed(SBMi_asyn)(ini_resource=spp_resource[sp] * 1. / 1e6,  # mol N/L
                                                ini_density=[spp_abundance[sp]],  # cells/L
                                                min_cell_size=[init_min_size[sp]],  # um^3
                                                max_cell_size=[init_max_size[sp]],  # um^3
                                                spp_names=[sp_short_names[sp]],
                                                dilution_rate=dilution,
                                                volume=volume,
                                                nsi_spp=sbmi_nsispp,
                                                nsi_min=sbmi_nsimin,
                                                nsi_max=sbmi_nsimax,
                                                time_step=sbmi_ts,
                                                time_end=time_end
                                                )
            sbmi_asyn_out.append(sbmi_asyn)
        if sbmi_syn_variant:
            sbmi_syn = dask.delayed(SBMi_syn)(ini_resource=spp_resource[sp] * 1. / 1e6,  # mol N/L
                                              ini_density=[spp_abundance[sp]],  # cells/L
                                              min_cell_size=[init_min_size[sp]],  # um^3
                                              max_cell_size=[init_max_size[sp]],  # um^3
                                              spp_names=[sp_short_names[sp]],
                                              dilution_rate=dilution,
                                              volume=volume,
                                              nsi_spp=sbmi_nsispp,
                                              nsi_min=sbmi_nsimin,
                                              nsi_max=sbmi_nsimax,
                                              time_step=sbmi_ts,
                                              time_end=time_end
                                              )
            sbmi_syn_out.append(sbmi_syn)

    results = dask.delayed(save_to_file)(sbmc_out, sbmi_asyn_out, sbmi_syn_out)

    dask.compute(results)
    client.close()
    cluster.close()


def run_exp(rel_size_range, dilution=0.0, volume=1, time_end=20,
            sbmc_numsc=[1], sbmi_nsispp=[500], sbmi_nsimin=100,
            sbmi_nsimax=1000, sbmi_ts=1 / (24 * 60),
            fname_base='Single_spp_exp_relsizerange_',
            init_size=('mean', 'min')):
    """
    function to run and save the single species simulation experiments.
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
    :param fname_base: string
        base of the output file name
    :param init_size: string
        whether to use minimum (min) or mean (mean) cell size to initialize species
        as reported by Marañón et al. (2013 Eco. Lett.)
    :return: ncfile with results of the simulations
    """

    start_comp_time = time.time()
    start_datetime = datetime.now()
    out_filename = fname_base + str(rel_size_range).zfill(2) + '.zarr'

    print('Compute single species experiments for %.2f dilution rate simulations.' % dilution)
    (sbmc_exp, sbmi_exp), spp_names = simulations(rel_size_range=rel_size_range,
                                                  dilution=dilution, volume=volume, time_end=time_end,
                                                  sbmc_numsc=sbmc_numsc, sbmi_nsispp=sbmi_nsispp,
                                                  sbmi_nsimin=sbmi_nsimin, sbmi_nsimax=sbmi_nsimax, sbmi_ts=sbmi_ts,
                                                  init_size=init_size)
    agents_arr_shape = (len(sbmi_exp[0].time), sbmi_nsimax)
    out_xr = xr.Dataset(data_vars={
        'sbmi_agents_size': (['spp_num', 'time_sbmi', 'num_agents'],
                             da.from_array([exp.agents_size.reshape(agents_arr_shape) for exp in sbmi_exp])),
        'sbmi_agents_biomass': (['spp_num', 'time_sbmi', 'num_agents'],
                                da.from_array([exp.agents_biomass.reshape(agents_arr_shape) for exp in sbmi_exp])),
        'sbmi_agents_abundance': (['spp_num', 'time_sbmi', 'num_agents'],
                                  da.from_array([exp.agents_abundance.reshape(agents_arr_shape) for exp in sbmi_exp])),
        'sbmi_agents_growth': (['spp_num', 'time_sbmi', 'num_agents'],
                               da.from_array([exp.agents_growth.reshape(agents_arr_shape) for exp in sbmi_exp])),
        'sbmi_resource': (['spp_num', 'time_sbmi'], da.from_array([exp.resource for exp in sbmi_exp])),
        'sbmi_tot_abundance': (['spp_num', 'time_sbmi'], da.from_array([exp.abundance for exp in sbmi_exp])),
        'sbmi_tot_biomass': (['spp_num', 'time_sbmi'], da.from_array([exp.biomass for exp in sbmi_exp])),
        'sbmi_tot_quota': (['spp_num', 'time_sbmi'], da.from_array([exp.quota for exp in sbmi_exp])),
        'sbmi_massbalance': (['spp_num', 'time_sbmi'], da.from_array([exp.massbalance for exp in sbmi_exp])),
        'sbmc_size': (['spp_num', 'num_spp_sc'], da.from_array([exp._size_range for exp in sbmc_exp])),
        'sbmc_biomass': (['spp_num', 'time_sbmc', 'num_spp_sc'], da.from_array([exp.biomass for exp in sbmc_exp])),
        'sbmc_abundance': (['spp_num', 'time_sbmc', 'num_spp_sc'], da.from_array([exp.abundance for exp in sbmc_exp])),
        'sbmc_quota': (['spp_num', 'time_sbmc', 'num_spp_sc'], da.from_array([exp.quota for exp in sbmc_exp])),
        'sbmc_growth': (['spp_num', 'time_sbmc', 'num_spp_sc'], da.from_array([exp.mus for exp in sbmc_exp])),
        'sbmc_resource': (['spp_num', 'time_sbmc'], da.from_array([exp.resource for exp in sbmc_exp]))
    }, coords={
        'spp_num': np.arange(22),
        'spp_names': (['spp_num'], np.array(spp_names)),
        'time_sbmi': sbmi_exp[0].time,
        'num_agents': np.arange(sbmi_nsimax),
        'time_sbmc': sbmc_exp[0].time,
        'num_spp_sc': np.arange(sbmc_numsc[0]),
    })
    out_xr.attrs['title'] = 'Single species experiments'
    out_xr.attrs['description'] = 'Single species experiments for ' + str(dilution) + \
                                  '% dilution rate and by using two size-based model types. ' \
                                  'One model type based on individuals (SBMi) ' \
                                  'and another type based on size classes (SBMc)'
    out_xr.attrs['simulations setup'] = 'relative size range:' + str(rel_size_range) + \
                                        ', dilution rate:' + str(dilution) + \
                                        ', volume:' + str(volume) + \
                                        ', maximum time of simulations:' + str(time_end) + \
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
    print('Save output as ncfile')
    with ProgressBar(), dask.config.set(scheduler='threads'):
        out_xr.to_zarr(out_filename)
    print('Completed! Total computation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))
