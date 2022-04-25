from re import search
import pkg_resources
import numpy as np
import pandas as pd
from itertools import combinations
from insidephy.size_based_models.sbm import SBMc, SBMi_asyn, SBMi_syn
import dask
from dask.distributed import Client, LocalCluster


def multiple_spp_sims(num_spp_exp, rel_size_range=0.25, dilution=0.0, volume=1.0, time_end=20,
                      sbmc_numsc=[10], sbmi_nsispp=[101], sbmi_nsimin=100, sbmi_nsimax=1000, sbmi_ts=1 / 24,
                      sbmc_variant=False, sbmi_asyn_variant=False, sbmi_syn_variant=False,
                      n_procs=6, n_threads=1, mem_lim=10e9, outfname_base='Multiple_spp_exp_'):
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

    def save_to_file(sbmc, sbmi_asyn, sbmi_syn, fname_base):
        if sbmc_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmc])
             .to_xarray()
             .to_zarr(fname_base + '.zarr', mode='w', group='sbmc')
             )
        if sbmi_asyn_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmi_asyn])
             .to_xarray()
             .to_zarr(fname_base + '.zarr', mode='w', group='sbmi_asyn')
             )
        if sbmi_syn_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmi_syn])
             .to_xarray()
             .to_zarr(fname_base + '.zarr', mode='w', group='sbmi_syn')
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

    results = dask.delayed(save_to_file)(sbmc_out, sbmi_asyn_out, sbmi_syn_out, outfname_base)

    dask.compute(results)
    client.close()
    cluster.close()


if __name__ == "__main__":
    multiple_spp_sims(2, rel_size_range=0, sbmc_numsc=[1], sbmi_ts=0.01,
                      sbmc_variant=True, sbmi_asyn_variant=True, outfname_base='Two_spp_exp_0percent')
    multiple_spp_sims(2, rel_size_range=0.25, sbmc_numsc=[10], sbmi_ts=0.01,
                      sbmc_variant=True, sbmi_asyn_variant=True, outfname_base='Two_spp_exp_25percent')
    multiple_spp_sims(2, rel_size_range=0.50, sbmc_numsc=[10], sbmi_ts=0.01,
                      sbmc_variant=True, sbmi_asyn_variant=True, outfname_base='Two_spp_exp_50percent')
    multiple_spp_sims(2, rel_size_range=0.75, sbmc_numsc=[10], sbmi_ts=0.01,
                      sbmc_variant=True, sbmi_asyn_variant=True, outfname_base='Two_spp_exp_75percent')
