import numpy as np
import matplotlib.pyplot as plt
from insidephy.size_based_models.sbm import SBMc, SBMi_syn, SBMi_asyn
import matplotlib.lines as mlines
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar


def sim_run(ini_resource=0.0001, ini_density=(1e4, 1e4), min_size=(1.5e1, 1.5e4), max_size=(2.5e1, 2.5e4),
            spp_names=('Aa', 'Bb'), dilution_rate=0.0, volume=1.0, nsi_spp=(500, 500), nsi_min=100,
            nsi_max=1900, num_sc=(50, 50), time_end=30, time_step=0.01, print_time_step=1,
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
    sbmc = dask.delayed(SBMc)(ini_resource=ini_resource, ini_density=ini_density, min_cell_size=min_size,
                              max_cell_size=max_size,
                              spp_names=spp_names, num_sc=num_sc, time_end=time_end,
                              dilution_rate=dilution_rate, volume=volume)
    sbmi_async = dask.delayed(SBMi_asyn)(ini_resource=ini_resource, ini_density=ini_density, min_cell_size=min_size,
                                         max_cell_size=max_size,
                                         spp_names=spp_names, nsi_spp=nsi_spp, nsi_min=nsi_min, nsi_max=nsi_max,
                                         volume=volume,
                                         time_step=time_step, time_end=time_end, print_time_step=print_time_step,
                                         dilution_rate=dilution_rate)
    sbmi_sync = dask.delayed(SBMi_syn)(ini_resource=ini_resource, ini_density=ini_density, min_cell_size=min_size,
                                       max_cell_size=max_size,
                                       spp_names=spp_names, nsi_spp=nsi_spp, nsi_min=nsi_min, nsi_max=nsi_max,
                                       volume=volume,
                                       time_step=time_step, time_end=time_end, print_time_step=print_time_step,
                                       dilution_rate=dilution_rate)
    sbmi_rand = dask.delayed(SBMi_asyn)(ini_resource=ini_resource, ini_density=ini_density, min_cell_size=min_size,
                                        max_cell_size=max_size,
                                        spp_names=spp_names, nsi_spp=nsi_spp, nsi_min=nsi_min, nsi_max=nsi_max,
                                        volume=volume,
                                        time_step=time_step, time_end=time_end, print_time_step=print_time_step,
                                        dilution_rate=dilution_rate, reproduction_random=True)

    sbm_out.append(sbmc)
    sbm_out.append(sbmi_async)
    sbm_out.append(sbmi_sync)
    sbm_out.append(sbmi_rand)

    with ProgressBar(), dask.config.set(scheduler='processes'):
        output = dask.compute(sbm_out)

    client.close()
    cluster.close()
    return output


def plots(dilution_rate=0.0, fname='sbmi_syn_rand_test.png'):
    ([sbmc, sbmi, sbmi_s, sbmi_r],) = sim_run(dilution_rate=dilution_rate)

    cols = ['#5494aeff', '#7cb950ff']

    fig1, axs1 = plt.subplots(5, 3, sharex='col', sharey='row', figsize=(10, 8))
    # Resources plots
    axs1[0, 0].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby('time').resource.first() * 1e3, c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[0, 0].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby('time').resource.first() * 1e3, c='grey', lw=3.0, alpha=0.5)
    axs1[0, 0].plot(sbmi_s.dtf.time.unique(), sbmi_s.dtf.groupby('time').resource.first() * 1e3, c='red', lw=3.0,
                    alpha=0.9)
    axs1[0, 0].plot(sbmi_r.dtf.time.unique(), sbmi_r.dtf.groupby('time').resource.first() * 1e3, ls=':', c='blue',
                    lw=3.0, alpha=0.9)
    # Quota plots
    axs1[1, 0].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby('time').apply(lambda x: np.sum(x.quota * x.abundance) * 1e3),
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 1].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.quota * x.abundance) * 1e3).loc[:, 'Aa'],
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 2].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.quota * x.abundance) * 1e3).loc[:, 'Bb'],
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 0].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby('time').quota.sum() * 1e3,
                    c='grey', lw=3.0, alpha=0.5)
    axs1[1, 1].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby(['time', 'spp']).quota.sum().loc[:, 'Aa'] * 1e3,
                    c='grey', lw=3.0, alpha=0.5)
    axs1[1, 2].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby(['time', 'spp']).quota.sum().loc[:, 'Bb'] * 1e3,
                    c='grey', lw=3.0, alpha=0.5)
    axs1[1, 0].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby('time').apply(lambda x: np.sum(x.quota * x.biomass * x.rep_nind)) * 1e3,
                    c='red', lw=3.0, alpha=0.9)
    axs1[1, 1].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.quota * x.biomass * x.rep_nind)).loc[:,
                    'Aa'] * 1e3,
                    c='red', lw=3.0, alpha=0.9)
    axs1[1, 2].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.quota * x.biomass * x.rep_nind)).loc[:,
                    'Bb'] * 1e3,
                    c='red', lw=3.0, alpha=0.9)
    axs1[1, 0].plot(sbmi_r.dtf.time.unique(),
                    sbmi_r.dtf.groupby('time').quota.sum() * 1e3,
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    axs1[1, 1].plot(sbmi_r.dtf.time.unique(),
                    sbmi_r.dtf.groupby(['time', 'spp']).quota.sum().loc[:, 'Aa'] * 1e3,
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    axs1[1, 2].plot(sbmi_r.dtf.time.unique(),
                    sbmi_r.dtf.groupby(['time', 'spp']).quota.sum().loc[:, 'Bb'] * 1e3,
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    # Abundance plots
    axs1[2, 0].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby('time').apply(lambda x: np.sum(x.abundance)),
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 1].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.abundance)).loc[:, 'Aa'],
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 2].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.abundance)).loc[:, 'Bb'],
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 0].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby('time').rep_nind.sum(), c='grey', lw=3.0, alpha=0.5)
    axs1[2, 1].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby(['time', 'spp']).rep_nind.sum().loc[:, 'Aa'], c='grey', lw=3.0, alpha=0.5)
    axs1[2, 2].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby(['time', 'spp']).rep_nind.sum().loc[:, 'Bb'], c='grey', lw=3.0, alpha=0.5)
    axs1[2, 0].plot(sbmi_s.dtf.time.unique(), sbmi_s.dtf.groupby('time').rep_nind.sum(), c='red', lw=3.0, alpha=0.9)
    axs1[2, 1].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby(['time', 'spp']).rep_nind.sum().loc[:, 'Aa'], c='red', lw=3.0, alpha=0.9)
    axs1[2, 2].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby(['time', 'spp']).rep_nind.sum().loc[:, 'Bb'], c='red', lw=3.0, alpha=0.9)
    axs1[2, 0].plot(sbmi_r.dtf.time.unique(),
                    sbmi_r.dtf.groupby('time').rep_nind.sum(),
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    axs1[2, 1].plot(sbmi_r.dtf.time.unique(),
                    sbmi_r.dtf.groupby(['time', 'spp']).rep_nind.sum().loc[:, 'Aa'],
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    axs1[2, 2].plot(sbmi_r.dtf.time.unique(),
                    sbmi_r.dtf.groupby(['time', 'spp']).rep_nind.sum().loc[:, 'Bb'],
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    # Biomass plots
    axs1[3, 0].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby('time').apply(lambda x: np.sum(x.biomass)),
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 1].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.biomass)).loc[:, 'Aa'],
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 2].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.biomass)).loc[:, 'Bb'],
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 0].plot(sbmi.dtf.groupby('time').time.first(),
                    sbmi.dtf.groupby('time').biomass.sum(),
                    c='grey', lw=3.0, alpha=0.5)
    axs1[3, 1].plot(sbmi.dtf.groupby('time').time.first(),
                    sbmi.dtf.groupby(['time', 'spp']).biomass.sum().loc[:, 'Aa'],
                    c='grey', lw=3.0, alpha=0.5)
    axs1[3, 2].plot(sbmi.dtf.groupby('time').time.first(),
                    sbmi.dtf.groupby(['time', 'spp']).biomass.sum().loc[:, 'Bb'],
                    c='grey', lw=3.0, alpha=0.5)
    axs1[3, 0].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby('time').apply(lambda x: np.sum(x.biomass * x.rep_nind)),
                    c='red', lw=3.0, alpha=0.9)
    axs1[3, 1].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.biomass * x.rep_nind)).loc[:, 'Aa'],
                    c='red', lw=3.0, alpha=0.9)
    axs1[3, 2].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.biomass * x.rep_nind)).loc[:, 'Bb'],
                    c='red', lw=3.0, alpha=0.9)
    axs1[3, 0].plot(sbmi_r.dtf.groupby('time').time.first(),
                    sbmi_r.dtf.groupby('time').biomass.sum(),
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    axs1[3, 1].plot(sbmi_r.dtf.groupby('time').time.first(),
                    sbmi_r.dtf.groupby(['time', 'spp']).biomass.sum().loc[:, 'Aa'],
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    axs1[3, 2].plot(sbmi_r.dtf.groupby('time').time.first(),
                    sbmi_r.dtf.groupby(['time', 'spp']).biomass.sum().loc[:, 'Bb'],
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    # Mean cell size plots
    axs1[4, 0].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby('time').apply(lambda x: np.sum(x.cell_size * x.abundance) / np.sum(x.abundance)),
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[4, 1].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby(['time', 'spp']).apply(
                        lambda x: np.sum(x.cell_size * x.abundance) / np.sum(x.abundance)).loc[:, 'Aa'],
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[4, 2].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby(['time', 'spp']).apply(
                        lambda x: np.sum(x.cell_size * x.abundance) / np.sum(x.abundance)).loc[:, 'Bb'],
                    c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[4, 0].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby('time').apply(lambda x: np.sum(x.cell_size * x.rep_nind) / np.sum(x.rep_nind)),
                    c='grey', lw=3.0, alpha=0.5)
    axs1[4, 1].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.cell_size * x.rep_nind) /
                                                                      np.sum(x.rep_nind)).loc[:, 'Aa'],
                    c='grey', lw=3.0, alpha=0.5)
    axs1[4, 2].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.cell_size * x.rep_nind) /
                                                                      np.sum(x.rep_nind)).loc[:, 'Bb'],
                    c='grey', lw=3.0, alpha=0.5)
    axs1[4, 0].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby('time').apply(lambda x: np.sum(x.cell_size * x.rep_nind) / np.sum(x.rep_nind)),
                    c='red', lw=3.0, alpha=0.9)
    axs1[4, 1].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.cell_size * x.rep_nind) /
                                                                        np.sum(x.rep_nind)).loc[:, 'Aa'],
                    c='red', lw=3.0, alpha=0.9)
    axs1[4, 2].plot(sbmi_s.dtf.time.unique(),
                    sbmi_s.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.cell_size * x.rep_nind) /
                                                                        np.sum(x.rep_nind)).loc[:, 'Bb'],
                    c='red', lw=3.0, alpha=0.9)

    axs1[4, 0].plot(sbmi_r.dtf.time.unique(),
                    sbmi_r.dtf.groupby('time').apply(lambda x: np.sum(x.cell_size * x.rep_nind) / np.sum(x.rep_nind)),
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    axs1[4, 1].plot(sbmi_r.dtf.time.unique(),
                    sbmi_r.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.cell_size * x.rep_nind) /
                                                                      np.sum(x.rep_nind)).loc[:, 'Aa'],
                    ls=':', c='blue', lw=3.0, alpha=0.9)
    axs1[4, 2].plot(sbmi_r.dtf.time.unique(),
                    sbmi_r.dtf.groupby(['time', 'spp']).apply(lambda x: np.sum(x.cell_size * x.rep_nind) /
                                                                      np.sum(x.rep_nind)).loc[:, 'Bb'],
                    ls=':', c='blue', lw=3.0, alpha=0.9)

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
    blackline = mlines.Line2D([], [], c='black', ls='--', lw=3.0)
    greyline = mlines.Line2D([], [], c='grey', lw=3.0)
    redline = mlines.Line2D([], [], c='red', lw=3.0)
    blueline = mlines.Line2D([], [], c='blue', ls=':', lw=3.0)
    axs1[0, 0].legend([blackline, greyline, redline, blueline], ['SBMc', 'SBMi', 'SBMi-s', 'SBMi-r'],
                      loc='lower left', prop={'size': 8})
    fig1.savefig(fname, dpi=600)


if __name__ == "__main__":
    plots()
