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

dask.config.set(scheduler='processes')


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


def run_exp(num_spp_exp, rel_size_range, dilution, volume, max_time,
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
            ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], da.from_array([exp.agents_abundance for exp in sbmi_exp])),
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


"""(sbmc_exp, sbmi_exp), spp_names = simulations(num_spp_exp=2, rel_size_range=0.25,
                                                  dilution=0.0, volume=1.0, max_time=10,
                                                  sbmc_numsc=2*[10], nsispp=2*[11],
                                                  sbmi_nsimin=10, sbmi_nsimax=100+10*2, sbmi_ts=1/2)"""

"""run_exp(num_spp_exp=2, rel_size_range=0.25,
           dilution=0.0, volume=1.0, max_time=10,
           sbmc_numsc=2*[10], sbmi_nsispp=2*[11],
           sbmi_nsimin=10, sbmi_nsimax=100+10*2, sbmi_ts=1/2)
"""