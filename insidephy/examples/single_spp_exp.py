from insidephy.size_based_models.sbm import SBMc, SBMi_asyn, SBMi_syn
import pandas as pd
import dask
from re import search
import pkg_resources
from dask.distributed import Client, LocalCluster


def single_spp_sims(rel_size_range, dilution=0.0, volume=1, time_end=20,
                    sbmc_numsc=[1], sbmi_nsispp=[500], sbmi_nsimin=100,
                    sbmi_nsimax=1000, sbmi_ts=1 / 24, init_size=('mean', 'min'),
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
             .to_zarr('Single_spp_exp_' + str(int(rel_size_range*100)) + 'percent.zarr', group='sbmc', mode='w')
             )
        if sbmi_asyn_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmi_asyn])
             .to_xarray()
             .to_zarr('Single_spp_exp_' + str(int(rel_size_range*100)) + 'percent.zarr', group='sbmi_asyn', mode='w')
             )
        if sbmi_syn_variant:
            (pd.concat([exp.dtf.assign(exp='-'.join(exp._params['spp_names'])) for exp in sbmi_syn])
             .to_xarray()
             .to_zarr('Single_spp_exp_' + str(int(rel_size_range*100)) + 'percent.zarr', group='sbmi_syn', mode='w')
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

    outputs = dask.delayed(save_to_file)(sbmc_out, sbmi_asyn_out, sbmi_syn_out)
    dask.compute(outputs)
    client.close()
    cluster.close()


if __name__ == "__main__":
    single_spp_sims(0, sbmc_variant=True, sbmi_ts=0.01, sbmi_asyn_variant=True, init_size='mean')
    single_spp_sims(0.25, sbmc_numsc=[10], sbmi_ts=0.01, sbmc_variant=True, sbmi_asyn_variant=True, init_size='mean')
    single_spp_sims(0.50, sbmc_numsc=[10], sbmi_ts=0.01, sbmc_variant=True, sbmi_asyn_variant=True, init_size='mean')
    single_spp_sims(0.75, sbmc_numsc=[10], sbmi_ts=0.01, sbmc_variant=True, sbmi_asyn_variant=True, init_size='mean')
