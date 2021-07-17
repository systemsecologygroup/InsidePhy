from insidephy.size_based_models.SBMc import SBMc
from insidephy.size_based_models.SBMi import SBMi
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

dask.config.set(scheduler='processes')


def simulations(rel_size_range, dilution, volume, sbmc_numsc, max_time, sbmi_nsispp,
                sbmi_nsimin, sbmi_nsimax, sbmi_ts, init_size=('mean', 'min')):
    data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    allometries = pd.read_hdf(data_path, 'allodtf')
    cultures = pd.read_hdf(data_path, 'batchdtf')
    cellsize = pd.read_hdf(data_path, 'sizedtf')

    spp_list = list(cultures.groupby('Species').groups.keys())
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
    sbmi_out = []
    for sp in range(len(spp_list)):
        sbmc = dask.delayed(SBMc)(ini_resource=spp_resource[sp] * 1. / 1e6,  # mol N/L
                                  ini_density=[spp_abundance[sp]],  # cells/L
                                  min_size=[init_min_size[sp]],  # um^3
                                  max_size=[init_max_size[sp]],  # um^3
                                  spp_names=[sp_short_names[sp]],
                                  dilution_rate=dilution,
                                  volume=volume,
                                  num_sc=sbmc_numsc,
                                  time_end=max_time,
                                  timeit=False,
                                  vectorize=True
                                  )
        sbmc_out.append(sbmc)

        sbmi = dask.delayed(SBMi)(ini_resource=spp_resource[sp] * 1. / 1e6,  # mol N/L
                                  ini_density=[spp_abundance[sp]],  # cells/L
                                  min_size=[init_min_size[sp]],  # um^3
                                  max_size=[init_max_size[sp]],  # um^3
                                  spp_names=[sp_short_names[sp]],
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

    with ProgressBar(), dask.config.set(scheduler='processes'):
        results = dask.compute(sbmc_out, sbmi_out)
    return results, sp_short_names


def run_exp(rel_size_range, dilution, volume, max_time,
            sbmc_numsc, sbmi_nsispp, sbmi_nsimin, sbmi_nsimax, sbmi_ts,
            fname_base='Single_spp_exp_relsizerange_',
            init_size=('mean', 'min')):
    start_comp_time = time.time()
    start_datetime = datetime.now()
    out_filename = fname_base + str(rel_size_range).zfill(2) + '.nc'

    print('Compute single species experiments for %.2f dilution rate simulations.' % dilution)
    (sbmc_exp, sbmi_exp), spp_names = simulations(rel_size_range=rel_size_range,
                                                  dilution=dilution, volume=volume, max_time=max_time,
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
        'sbmc_size': (['spp_num', 'num_spp_sc'], da.from_array([exp.size_range for exp in sbmc_exp])),
        'sbmc_biomass': (['spp_num', 'time_sbmc','num_spp_sc'], da.from_array([exp.biomass for exp in sbmc_exp])),
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
    print('Save output as ncfile')
    with ProgressBar(), dask.config.set(scheduler='threads'):
        out_xr.to_netcdf(out_filename)
    print('Completed! Total computation time %.2f minutes' % ((time.time() - start_comp_time) / 60.))
