import numpy as np
import matplotlib.pyplot as plt
from insidephy.size_based_models.SBMi import SBMi
from insidephy.size_based_models.SBMc import SBMc
from insidephy.size_based_models.trait_metrics import cwm
import matplotlib.lines as mlines
import dask
from dask.distributed import Client, LocalCluster


def sim_run(ini_resource=0.0002, ini_density=(1e4, 1e4), min_size=(1.5e1, 1.5e4), max_size=(2.5e1, 2.5e4),
            spp_names=('Aa', 'Bb'), dilution_rate=0.0, volume=1.0, nsi_spp=(500, 500), nsi_min=200,
            nsi_max=2000, num_sc=(100, 100), time_end=30, time_step=1 / 24, print_time_step=1,
            n_procs=2, n_threads=1, mem_lim=2e9):
    """
    Simulations of a minimal running example using two size based model types with two arbitrary species.
    The computation is parallelized using the Dask library as a demonstration but it is not necessary.
    :param ini_resource: float
        initial value resource concentration in units of nitrogen (M N)
    :param ini_density: list or tuple of floats
        initial density of cells in number of cells per litre
    :param min_size: list or tuple of floats
        minimum cell size per species or strain
    :param max_size: list or tuple of floats
        maximum cell size per species or strain
    :param spp_names: list or tuple of floats
        two letter name tag per species or strain
    :param dilution_rate: float
        rate of medium exchange in the culture system
    :param volume: float
        volume of the culture system
    :param nsi_spp: list or tuple of integers
        number of super individuals per species or strain used in SBMi model
    :param nsi_min: integer
        minimum number of super individual for all species and strain used in SBMi
    :param nsi_max: integer
        maximum number of super individual for all species and strain used in SBMi
    :param num_sc: list or tuple of integers
        number of size classes per species or strain used in SBMc
    :param time_end: integer
        final time in days used in the simulations
    :param time_step: float
        time steps used in the SBMi
    :param print_time_step: integer
        time step in days used to store results in SBMi
    :param n_procs: integer
        number of cpus to compute the simulations
    :param n_threads:
        number of threads per cpu to use in the computations
    :param mem_lim:
        maximum memory usage per cpu
    :return: tuple of objects
        results of the simulations of SBMc and SBMi
    """
    cluster = LocalCluster(n_workers=n_procs, threads_per_worker=n_threads, memory_limit=mem_lim)
    client = Client(cluster)

    sbm_out = []
    sbmc = dask.delayed(SBMc)(ini_resource=ini_resource, ini_density=ini_density, min_size=min_size, max_size=max_size,
                              spp_names=spp_names, num_sc=num_sc, time_end=time_end,
                              dilution_rate=dilution_rate, volume=volume)
    sbmi = dask.delayed(SBMi)(ini_resource=ini_resource, ini_density=ini_density, min_size=min_size, max_size=max_size,
                              spp_names=spp_names, nsi_spp=nsi_spp, nsi_min=nsi_min, nsi_max=nsi_max, volume=volume,
                              time_step=time_step, time_end=time_end, print_time_step=print_time_step,
                              dilution_rate=dilution_rate)
    sbm_out.append(sbmc)
    sbm_out.append(sbmi)

    output = dask.compute(sbm_out)
    client.close()
    cluster.close()
    return output


def plots():
    ([sbmc, sbmi],) = sim_run()

    cols = ['#5494aeff', '#7cb950ff']

    fig1, axs1 = plt.subplots(5, 3, sharex='col', sharey='row', figsize=(10, 8))
    # Resources plots
    axs1[0, 0].plot(sbmc.time, sbmc.resource * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[0, 0].plot(sbmi.time, sbmi.resource * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    # Quota plots
    axs1[1, 0].plot(sbmc.time, np.sum(sbmc.quota * sbmc.abundance, axis=1) * 1e3,
                    c='black', lw=3.0, alpha=0.9)
    axs1[1, 0].plot(sbmi.time, sbmi.quota * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 1].plot(sbmc.time, (sbmc.quota[:, :sbmc.numsc[0]] *
                                sbmc.abundance[:, :sbmc.numsc[0]]).sum(axis=1) * 1e3,
                    c=cols[0], lw=3.0, alpha=0.9)
    axs1[1, 1].plot(sbmi.time, sbmi.agents_quota.sum(axis=2)[0, :] * 1e3, c=cols[0], ls='--', lw=3.0, alpha=0.5)
    axs1[1, 2].plot(sbmc.time, (sbmc.quota[:, sbmc.numsc[0]:] *
                                sbmc.abundance[:, sbmc.numsc[0]:]).sum(axis=1) * 1e3,
                    c=cols[1], lw=3.0, alpha=0.9)
    axs1[1, 2].plot(sbmi.time, sbmi.agents_quota.sum(axis=2)[1, :] * 1e3, c=cols[1], ls='--', lw=3.0, alpha=0.5)
    # Abundance plots
    axs1[2, 0].plot(sbmc.time, np.sum(sbmc.abundance[:, :sbmc.numsc[0]], axis=1), c='black', lw=3.0, alpha=0.9)
    axs1[2, 0].plot(sbmi.time, sbmi.abundance, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 1].plot(sbmc.time, sbmc.abundance[:, :sbmc.numsc[0]].sum(axis=1), c=cols[0], lw=3.0, alpha=0.9)
    axs1[2, 1].plot(sbmi.time, sbmi.agents_abundance.sum(axis=2)[0, :], c='black', ls='--', lw=3.0, alpha=0.5)
    axs1[2, 2].plot(sbmc.time, sbmc.abundance[:, sbmc.numsc[0]:].sum(axis=1), c=cols[1], lw=3.0, alpha=0.9)
    axs1[2, 2].plot(sbmi.time, sbmi.agents_abundance.sum(axis=2)[1, :], c='black', ls='--', lw=3.0, alpha=0.5)
    # Biomass plots
    axs1[3, 0].plot(sbmc.time, np.sum(sbmc.biomass[:, :sbmc.numsc[0]], axis=1), c='black', lw=3.0, alpha=0.9)
    axs1[3, 0].plot(sbmi.time, sbmi.biomass, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 1].plot(sbmc.time, sbmc.biomass[:, :sbmc.numsc[0]].sum(axis=1), c=cols[0], lw=3.0, alpha=0.9)
    axs1[3, 1].plot(sbmi.time, sbmi.agents_biomass.sum(axis=2)[0, :], c='black', ls='--', lw=3.0, alpha=0.5)
    axs1[3, 2].plot(sbmc.time, sbmc.biomass[:, sbmc.numsc[0]:].sum(axis=1), c=cols[1], lw=3.0, alpha=0.9)
    axs1[3, 2].plot(sbmi.time, sbmi.agents_biomass.sum(axis=2)[1, :], c='black', ls='--', lw=3.0, alpha=0.5)
    # Mean cell size plots
    axs1[4, 0].plot(sbmc.time, cwm(np.broadcast_to(sbmc.size_range, (100, 200)), sbmc.abundance),
                    c='black', lw=3.0, alpha=0.9)
    axs1[4, 0].plot(sbmi.time, cwm(sbmi.agents_size.reshape(31, 4000), sbmi.agents_abundance.reshape(31, 4000)),
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[4, 1].plot(sbmc.time, cwm(np.broadcast_to(sbmc.size_range[:sbmc.numsc[0]], (100, 100)),
                                   sbmc.abundance[:, :sbmc.numsc[0]]), c=cols[0], lw=3.0, alpha=0.9)
    axs1[4, 1].plot(sbmi.time, cwm(sbmi.agents_size[0, :, :], sbmi.agents_abundance[0, :, :]), c=cols[0],
                    ls='--', lw=3.0, alpha=0.5)
    axs1[4, 2].plot(sbmc.time, cwm(np.broadcast_to(sbmc.size_range[sbmc.numsc[0]:], (100, 100)),
                                   sbmc.abundance[:, sbmc.numsc[0]:]), c=cols[1], lw=3.0, alpha=0.9)
    axs1[4, 2].plot(sbmi.time, cwm(sbmi.agents_size[1, :, :], sbmi.agents_abundance[1, :, :]), c=cols[1],
                    ls='--', lw=3.0, alpha=0.5)
    # customization
    axs1[2, 0].set_yscale('log')
    axs1[3, 0].set_yscale('log')
    axs1[4, 0].set_yscale('log')
    axs1[0, 0].set_xlim(0, 30)
    axs1[0, 1].set_xlim(0, 30)
    axs1[0, 2].set_xlim(0, 30)
    axs1[0, 1].axis('off')
    axs1[0, 2].axis('off')
    axs1[0, 0].set_title("Total", weight='bold')
    axs1[1, 1].set_title("Aa", weight='bold')
    axs1[1, 2].set_title("Bb", weight='bold')
    axs1[0, 0].set_ylabel('Nutrients\n[mM N]', weight='bold')
    axs1[1, 0].set_ylabel('PON\n[mM N]', weight='bold')
    axs1[2, 0].set_ylabel('Abundance\n[cells L$^{-1}$]', weight='bold')
    axs1[3, 0].set_ylabel('Biomass\n[mM C]', weight='bold')
    axs1[4, 0].set_ylabel('Mean size\n[$\mu$m$^3$]', weight='bold')
    axs1[4, 0].set_xlabel('Time [days]', weight='bold')
    axs1[4, 1].set_xlabel('Time [days]', weight='bold')
    axs1[4, 2].set_xlabel('Time [days]', weight='bold')
    fig1.subplots_adjust(wspace=0.09)
    blackline = mlines.Line2D([], [], c='black', lw=3.0)
    greyline = mlines.Line2D([], [], c='grey', ls='--', lw=3.0)
    axs1[0, 0].legend([blackline, greyline], ['SBMc', 'SBMi'], loc='lower left')
    fig1.savefig('MREG.png', dpi=600)
