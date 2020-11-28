# import dask.dataframe as dd
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

# from dask.distributed import Client, progress

# cluster = LocalCluster(
#     #     n_workers=4,
#     #     ip='127.0.0.1',
# )
# client = Client(cluster)


# ==================================
# run this line in ipython console
# from dask.distributed import Client
# client = Client()
# client = Client(processes=False)
# dask.config.set(scheduler='processes')
# dask.config.set(scheduler='sync')
from dask.distributed import Client
# open in browser
# http://localhost:8787/status
# ==================================

#

#client = Client(processes=False)#, #threads_per_worker=4,
               # n_workers=1)

dask.config.set(scheduler='processes')


def simulations(num_spp_exp, rel_size_range, dilution, volume, sbmc_numsc, maxtime, nsispp,
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
        # print(i+1, 'of', len(spp_exps), spp_exps[i], sp_short_names[i])
        sbmc = dask.delayed(SBMc)(ini_resource=np.max(init_r[c]) * 1. / 1e6,  # mol N/L
                                  ini_density=init_d[c],  # cells/L
                                  minsize=init_min_size[c],  # um^3
                                  maxsize=init_max_size[c],  # um^3
                                  spp_names=sp_short_names[c],
                                  dilution_rate=dilution,
                                  volume=volume,
                                  numsc=sbmc_numsc,
                                  tend=maxtime,
                                  timeit=False,
                                  vectorize=True
                                  )
        sbmc_out.append(sbmc)
        # for i in range(len(spp_exps)):
        sbmi = dask.delayed(SBMi)(ini_resource=np.max(init_r[c]) * 1. / 1e6,  # mol N/L
                                  ini_density=init_d[c],  # cells/L
                                  minsize=init_min_size[c],  # um^3
                                  maxsize=init_max_size[c],  # um^3
                                  spp_names=sp_short_names[c],
                                  dilution_rate=dilution,
                                  volume=volume,
                                  nsi_spp=nsispp,
                                  nsi_min=sbmi_nsimin,
                                  nsi_max=sbmi_nsimax,
                                  time_step=sbmi_ts,
                                  time_end=maxtime,
                                  timeit=False
                                  )
        sbmi_out.append(sbmi)
    # dask.delayed()(sbmc_out)
    # dask.delayed()(sbmi_out), dask.compute(sbmi_out, scheduler='processes'),
    # lazy_out = dask.persist(*sbmc_out, *sbmi_out)
    # lazy_sbmc = dask.persist(sbmc_out, scheduler='processes')
    # lazy_sbmi = dask.persist(sbmi_out)
    # x = client.submit(sbmc_out)
    # return dask.compute(sbmc_out, scheduler='processes'), dask.compute(sbmi_out), sp_short_names #, lazy_sbmi
    return dask.compute(sbmc_out, sbmi_out), sp_short_names


def run_exp(num_spp_exp, rel_size_range, dilution, volume, maxtime,
            sbmc_numsc, sbmi_nsispp, sbmi_nsimin, sbmi_nsimax, sbmi_ts):
    # num_spp_exp = 22
    start_comp_time = time.time()
    start_datetime = datetime.now()
    out_filename = 'Multiple_spp_exp_' + str(num_spp_exp).zfill(2) + 'spp' + '.nc'

    """return simulations(num_spp_exp=num_spp_exp, rel_size_range=rel_size_range,
                                                  dilution=dilution, volume=volume, maxtime=maxtime,
                                                  sbmc_numsc=sbmc_numsc, nsispp=sbmi_nsispp,
                                                  sbmi_nsimin=sbmi_nsimin, sbmi_nsimax=sbmi_nsimax, sbmi_ts=sbmi_ts)
    """
    (sbmc_exp, sbmi_exp), spp_names = simulations(num_spp_exp=num_spp_exp, rel_size_range=rel_size_range,
                                                  dilution=dilution, volume=volume, maxtime=maxtime,
                                                  sbmc_numsc=sbmc_numsc, nsispp=sbmi_nsispp,
                                                  sbmi_nsimin=sbmi_nsimin, sbmi_nsimax=sbmi_nsimax, sbmi_ts=sbmi_ts)
    # dilution=dilution, volume=1.0, maxtime=20,
    # sbmc_numsc=[100] * num_spp_exp, nsispp=[500] * num_spp_exp,
    # sbmi_nsimin=100, sbmi_nsimax=1000, sbmi_ts=1 / (24*60))

    out_xr = xr.Dataset(data_vars={
        'sbmi_agents_size': (
            ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], np.array([exp.agents_size for exp in sbmi_exp])),
        'sbmi_agents_biomass': (
            ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], np.array([exp.agents_biomass for exp in sbmi_exp])),
        'sbmi_agents_abundance': (
            ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], np.array([exp.agents_abundance for exp in sbmi_exp])),
        'sbmi_agents_growth': (
            ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'], np.array([exp.agents_growth for exp in sbmi_exp])),
        'sbmi_resource': (['exp_num', 'time_sbmi'], np.array([exp.resource for exp in sbmi_exp])),
        'sbmi_tot_abundance': (['exp_num', 'time_sbmi'], np.array([exp.abundance for exp in sbmi_exp])),
        'sbmi_tot_biomass': (['exp_num', 'time_sbmi'], np.array([exp.biomass for exp in sbmi_exp])),
        'sbmi_tot_quota': (['exp_num', 'time_sbmi'], np.array([exp.quota for exp in sbmi_exp])),
        'sbmi_massbalance': (['exp_num', 'time_sbmi'], np.array([exp.massbalance for exp in sbmi_exp])),
        'sbmc_size': (['exp_num', 'idx_num_sc'], np.array([exp.size_range for exp in sbmc_exp])),
        'sbmc_biomass': (['exp_num', 'time_sbmc', 'idx_num_sc'], np.array([exp.biomass for exp in sbmc_exp])),
        'sbmc_abundance': (['exp_num', 'time_sbmc', 'idx_num_sc'], np.array([exp.abundance for exp in sbmc_exp])),
        'sbmc_quota': (['exp_num', 'time_sbmc', 'idx_num_sc'], np.array([exp.quota for exp in sbmc_exp])),
        'sbmc_growth': (['exp_num', 'time_sbmc', 'idx_num_sc'], np.array([exp.mus for exp in sbmc_exp])),
        'sbmc_resource': (['exp_num', 'time_sbmc'], np.array([exp.resource for exp in sbmc_exp])),

    }, coords={
        'exp_num': np.arange(len(sbmc_exp)),
        'exp_names': (['exp_num'], ['-'.join(spp_exp) for spp_exp in spp_names]),
        'num_spp_exp': np.arange(num_spp_exp),
        'spp_names': (['exp_num', 'num_spp_exp'], np.array(spp_names)),
        'time_sbmi': sbmi_exp[0].time,
        'time_sbmc': sbmc_exp[0].time,
        'num_agents': np.arange(sbmi_nsimax),
        'num_spp_sc': (['exp_num', 'num_spp_exp'], [exp.numsc for exp in sbmc_exp]),
        # 'idx_num_sc': np.arange(sum(sbmc_exp[0].numsc))
    })
    out_xr.attrs['title'] = 'Multiple species experiments'
    out_xr.attrs['description'] = 'Experiments for a combination of ' + str(num_spp_exp) + \
                                  ' species and by using two size-based model types. ' \
                                  'One model type based on individuals (SBMi) ' \
                                  'and another type based on size classes (SBMc)'
    out_xr.attrs['simulations setup'] = 'relative size range:' + str(rel_size_range) + \
                                        ', dilution rate:' + str(dilution) + \
                                        ', volume:' + str(volume) + \
                                        ', maximum time of simulations:' + str(maxtime) + \
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
    out_xr.attrs['total running time'] = ((time.time() - start_comp_time) / 60.)
    out_xr.to_netcdf(out_filename)
    print('Completed! Total time %.2f minutes' % ((time.time() - start_comp_time) / 60.))
    # return out_xr


# xarr1 = run_exp(1, 0.25, 0.0, 1.0, 20, [10]*1, [50]*1, 10*1, 100*1, 1/2)
# xarr2 = run_exp(2, 0.25, 0.0, 1.0, 20, [10]*2, [50]*2, 10*2, 100*2, 1/2)
# xarr22 = run_exp(22, 0.25, 0.0, 1.0, 20, [10]*22, [50]*22, 10*22, 100*22, 1/2)
# dtf1 = xarr1['sbmi_agents_size'][0].to_dataframe(dim_order=('spp_num', 'time_sbmi', 'num_agents'))
# dtf2 = xarr2['sbmi_agents_size'][0].to_dataframe(dim_order=('spp_num', 'time_sbmi', 'num_agents'))
# dtf22 = xarr22['sbmi_agents_size'][0].to_dataframe(dim_order=('spp_num', 'time_sbmi', 'num_agents'))

"""
def initial_conditions(num_spp_exp, rel_size_range):
    allometries = pd.read_hdf('maranon_2013EcoLet_data.h5', 'allodtf')
    cultures = pd.read_hdf('maranon_2013EcoLet_data.h5', 'batchdtf')

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

    return spp_exps, init_r, init_d, init_min_size, init_max_size, sp_short_names

"""

# (sbmc_exp, sbmi_exp), spp_names = simulations(num_spp_exp=1, rel_size_range=0.25, dilution=0.0, volume=1.0,
#                                               sbmc_numsc=[10], nsispp=[50], sbmi_nsimin=10, sbmi_nsimax=100,
#                                               sbmi_ts=1/24,maxtime=20)


"""def initial_dask(num_spp_exp):
    allometries = pd.read_hdf('maranon_2013EcoLet_data.h5', 'allodtf')
    allometries_dd = dd.from_pandas(allometries, 4)

    cultures = pd.read_hdf('maranon_2013EcoLet_data.h5', 'batchdtf')
    cultures_dd = dd.from_pandas(cultures, 4)

    spp_list = list(allometries_dd.Species.compute())
    spp_exps = list(combinations(spp_list, num_spp_exp))
    spp_resource = cultures_dd.groupby('Species').first().NO3_uM.compute()
    spp_abundance = cultures_dd.groupby('Species').first().Abun_cellmL.compute() * 1e3 / 1.

    init_r = [[spp_resource[sp] for sp in exp] for exp in spp_exps]
    init_d = [[spp_abundance[sp] for sp in exp] for exp in spp_exps]

    return init_r, init_d
"""

# r_dk, d_dk = initial_conditions(2)
# r_old, d_old = initial_old(2)


"""output = []
for exp in spp_exps:
    # print('Simulation for ' + '-'.join(exp))
    ini_r = delayed(np.max)([culbyspp_dtf.first().NO3_uM[sp] for sp in exp])
    ini_d = delayed()([culbyspp_dtf.first().Abun_cellmL[sp] * 1e3 / 1. for sp in exp])
    min_size = delayed()([allobyspp_dtf.first().Vcell[sp] - (allobyspp_dtf.first().Vcell[sp] * rel_size_range) for sp in exp])
    max_size = delayed()([allobyspp_dtf.first().Vcell[sp] + (allobyspp_dtf.first().Vcell[sp] * rel_size_range) for sp in exp])
    sppnames = delayed()([sp[0] + sp[search('_', sp).span()[1]] for sp in exp])
    # pftnames = [allobyspp_dtf.first().PFT[sp] for sp in exp]
    sbmc = delayed(SBMc)(ini_resource=ini_r * 1. / 1e6,  # mol N/L
                         ini_density=ini_d,  # cells/L
                         minsize=min_size,  # um^3
                         maxsize=max_size,  # um^3
                         spp_names=sppnames,
                         dilution_rate=dilution,
                         volume=volume,
                         numsc=sbmsc_numsc,
                         tend=maxtime)
    output.append(sbmc)
    sbmi_mod = delayed(SBMi)(ini_resource=ini_r * 1. / 1e6,  # mol N/L
                             ini_density=ini_d,  # cells/L
                             minsize=min_size,  # um^3
                             maxsize=max_size,  # um^3
                             spp_names=sppnames,
                             dilution_rate=dilution,
                             volume=volume,
                             nsi_spp=nsispp,
                             nsi_min=sbmind_nsimin,
                             nsi_max=sbmind_nsimax,
                             time_step=sbmind_ts,
                             time_end=maxtime)
return delayed()(output)
"""

"""def test_nodask():
    output = []
    for vec in [True, False]:
        sbmc = SBMc(ini_resource=0.0001, ini_density=[1e6], minsize=[500], maxsize=[1500],
                    spp_names=['Aa'], numsc=[100], tend=20, dilution_rate=0.0, volume=1.0, vectorize=vec)
        output.append(sbmc)
    return output


spp_exps, ini_r, ini_d, min_size, max_size, sppnames = initial_conditions(1, 0.25)


def test_dask():
    output = []
    for i in range(22):
        print(sppnames[i])
        sbmc = dask.delayed(SBMc)(ini_resource=0.0001, ini_density=[1e6], minsize=[500], maxsize=[1500],
                                  spp_names=sppnames[i], numsc=[100], tend=20, dilution_rate=0.0, volume=1.0,
                                  vectorize=True)
        output.append(sbmc)
    results = dask.compute(*output)
    return results  # delayed()(output)
# test_dask()
# test_nodk = test_nodask()
# test_dk = .compute()
# sims = simulations(1, 0.25, 20, 0.0, 1.0, 100).compute()
"""
