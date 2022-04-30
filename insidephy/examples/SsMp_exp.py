from insidephy.size_based_models.sbm import SBMc, SBMi_asyn, SBMi_syn
import pandas as pd
import pkg_resources
from re import search
import numpy as np
import dask
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.patches import Patch
import xarray as xr
import matplotlib.gridspec as gridspec


def sim_run(rel_size_range=0.25, ss_mp_names=['Synechococcus_sp', 'Micromonas_pusilla'],
            nsi_spp=[500, 500], nsi_min=100, nsi_max=1900,
            numsc=[10, 10], tend=20, dilution_rate=[0.0, 0.25, 0.50],
            time_step=0.01, volume=1.0, n_procs=6, n_threads=1, mem_lim=10e9,
            sbmc_variant=True, sbmi_asyn_variant=True
            ):
    """
    function to compute competition experiments between two species
    from the pool of species reported by Marañon et al (2013 Eco. Lett.)
    at three dilution rates
    :param rel_size_range: float
        initial size variability relative to its initial mean size
    :param ss_mp_names: list or tuple
        list with names of species as reported by Marañon et al. (2013 Eco. Lett.)
    :param nsi_spp: list or tuple
        list with initial number of super individuals to use in SBMi model type
    :param nsi_min: minimum number of super individuals
    :param nsi_max: maximum number of super individuals
    :param numsc: list with initial number of size classes use in SBMc model type
    :param tend: Total simulation time in days
    :param dilution_rate: list of floats
        List with three dilution rates
    :param time_step: time steps use in SBMi model type
    :param volume: volume of flask where species compete
    :return: tuple with two list containing the results for the SBMc and the SBMi model type
    """

    def save_to_file(sbmc, sbmi_asyn):
        if sbmc_variant:
            (pd.concat([exp.dtf.assign(exp='D' + str(exp._params['dilution_rate'])) for exp in sbmc])
             .to_xarray()
             .to_zarr('SsMp_exp.zarr', group='sbmc', mode='w')
             )
        if sbmi_asyn_variant:
            (pd.concat([exp.dtf.assign(exp='D' + str(exp._params['dilution_rate'])) for exp in sbmi_asyn])
             .to_xarray()
             .to_zarr('SsMp_exp.zarr', group='sbmi_asyn', mode='w')
             )

    cluster = LocalCluster(n_workers=n_procs, threads_per_worker=n_threads, memory_limit=mem_lim)
    client = Client(cluster)
    print(client.dashboard_link)

    data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    allometries = pd.read_hdf(data_path, 'allodtf')
    cultures = pd.read_hdf(data_path, 'batchdtf')
    spp_resource = cultures.groupby('Species').first().NO3_uM
    spp_abundance = cultures.groupby('Species').first().Abun_cellmL * 1e3 / 1.
    spp_size = allometries.groupby('Species').first().Vcell

    init_r = [spp_resource[sp] for sp in ss_mp_names]
    init_d = [spp_abundance[sp] for sp in ss_mp_names]
    init_min_size = [spp_size[sp] - (spp_size[sp] * rel_size_range) for sp in ss_mp_names]
    init_max_size = [spp_size[sp] + (spp_size[sp] * rel_size_range) for sp in ss_mp_names]
    sp_short_names = [sp[0] + sp[search('_', sp).span()[1]] for sp in ss_mp_names]

    sbmc_out = []
    sbmi_asyn_out = []
    for dr in dilution_rate:
        if sbmc_variant:
            sbmc = dask.delayed(SBMc)(ini_resource=np.max(init_r) * 1. / 1e6,  # mol N/L
                                      ini_density=init_d,  # cells/L
                                      min_cell_size=init_min_size,  # um^3
                                      max_cell_size=init_max_size,  # um^3
                                      spp_names=sp_short_names,
                                      dilution_rate=dr,
                                      volume=volume,
                                      num_sc=numsc,
                                      time_end=tend
                                      )
            sbmc_out.append(sbmc)
        if sbmi_asyn_variant:
            sbmi_asyn = dask.delayed(SBMi_asyn)(ini_resource=np.max(init_r) * 1. / 1e6,  # mol N/L
                                                ini_density=init_d,  # cells/L
                                                min_cell_size=init_min_size,  # um^3
                                                max_cell_size=init_max_size,  # um^3
                                                spp_names=sp_short_names,
                                                dilution_rate=dr,
                                                volume=volume,
                                                nsi_spp=nsi_spp,
                                                nsi_min=nsi_min,
                                                nsi_max=nsi_max,
                                                time_step=time_step,
                                                time_end=tend
                                                )
            sbmi_asyn_out.append(sbmi_asyn)

    results = dask.delayed(save_to_file)(sbmc_out, sbmi_asyn_out)
    output = dask.compute(results)
    client.close()
    cluster.close()


def temporal_dynamics_plot():
    """
    function to plot aggregate results of competition experiments for two species under three dilution rates
    :return: plot
    """
    Ss_Mp_data_path = pkg_resources.resource_filename('insidephy.examples', 'SsMp_exp.zarr')
    ds_sbmc = xr.open_zarr(Ss_Mp_data_path + '/sbmc').to_dataframe()
    ds_sbmi_asyn = xr.open_zarr(Ss_Mp_data_path + '/sbmi_asyn').to_dataframe()

    fig1, axs1 = plt.subplots(4, 3, sharex='col', sharey='row', figsize=(10, 8))
    # Resources plots
    axs1[0, 0].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).resource.first().loc['D0.0', :] * 1e3,
                    c='black', lw=3.0, alpha=0.9)
    axs1[0, 0].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).resource.first().loc['D0.0', :] * 1e3,
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[0, 1].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).resource.first().loc['D0.25', :] * 1e3,
                    c='black', lw=3.0, alpha=0.9)
    axs1[0, 1].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).resource.first().loc['D0.25', :] * 1e3,
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[0, 2].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).resource.first().loc['D0.5', :] * 1e3,
                    c='black', lw=3.0, alpha=0.9)
    axs1[0, 2].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).resource.first().loc['D0.5', :] * 1e3,
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    # Quota plots
    axs1[1, 0].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).apply(lambda x: np.sum(x.quota * x.abundance) * 1e3).loc['D0.0',
                    :],
                    c='black', lw=3.0, alpha=0.9)
    axs1[1, 0].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).quota.sum().loc['D0.0', :] * 1e3,
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 1].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).apply(lambda x: np.sum(x.quota * x.abundance) * 1e3).loc['D0.25',
                    :],
                    c='black', lw=3.0, alpha=0.9)
    axs1[1, 1].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).quota.sum().loc['D0.25', :] * 1e3,
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 2].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).apply(lambda x: np.sum(x.quota * x.abundance) * 1e3).loc['D0.5',
                    :],
                    c='black', lw=3.0, alpha=0.9)
    axs1[1, 2].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).quota.sum().loc['D0.5', :] * 1e3,
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    # Abundance plots
    axs1[2, 0].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).apply(lambda x: np.sum(x.abundance)).loc['D0.0', :],
                    c='black', lw=3.0, alpha=0.9)
    axs1[2, 0].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).rep_nind.sum().loc['D0.0', :],
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 1].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).apply(lambda x: np.sum(x.abundance)).loc['D0.25', :],
                    c='black', lw=3.0, alpha=0.9)
    axs1[2, 1].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).rep_nind.sum().loc['D0.25', :],
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 2].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).apply(lambda x: np.sum(x.abundance)).loc['D0.5', :],
                    c='black', lw=3.0, alpha=0.9)
    axs1[2, 2].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).rep_nind.sum().loc['D0.5', :],
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    # Biomass plots
    axs1[3, 0].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).apply(lambda x: np.sum(x.biomass)).loc['D0.0', :],
                    c='black', lw=3.0, alpha=0.9)
    axs1[3, 0].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).biomass.sum().loc['D0.0', :],
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 1].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).apply(lambda x: np.sum(x.biomass)).loc['D0.25', :],
                    c='black', lw=3.0, alpha=0.9)
    axs1[3, 1].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).biomass.sum().loc['D0.25', :],
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 2].plot(ds_sbmc.groupby('time').time.first(),
                    ds_sbmc.groupby(['exp', 'time']).apply(lambda x: np.sum(x.biomass)).loc['D0.5', :],
                    c='black', lw=3.0, alpha=0.9)
    axs1[3, 2].plot(ds_sbmi_asyn.time.unique(),
                    ds_sbmi_asyn.groupby(['exp', 'time']).biomass.sum().loc['D0.5', :],
                    c='grey', ls='--', lw=3.0, alpha=0.9)
    # customization
    axs1[2, 0].set_yscale('log')
    axs1[3, 0].set_yscale('log')
    axs1[0, 0].set_xlim(0, 20)
    axs1[0, 1].set_xlim(0, 20)
    axs1[0, 2].set_xlim(0, 20)
    axs1[0, 0].set_ylim(0, 0.2)
    axs1[1, 0].set_ylim(0, 0.2)
    axs1[0, 0].set_ylabel('Nutrients\n[mM N]', weight='bold')
    axs1[1, 0].set_ylabel('PON\n[mM N]', weight='bold')
    axs1[2, 0].set_ylabel('Abundance\n[cells L$^{-1}$]', weight='bold')
    axs1[3, 0].set_ylabel('Biomass\n[mM C]', weight='bold')
    axs1[0, 0].set_title('0% dilution rate', weight='bold')
    axs1[0, 1].set_title('25% dilution rate', weight='bold')
    axs1[0, 2].set_title('50% dilution rate', weight='bold')
    axs1[3, 0].set_xlabel('Time [days]', weight='bold')
    axs1[3, 1].set_xlabel('Time [days]', weight='bold')
    axs1[3, 2].set_xlabel('Time [days]', weight='bold')
    fig1.subplots_adjust(wspace=0.09)
    blackline = mlines.Line2D([], [], c='black', lw=3.0)
    greyline = mlines.Line2D([], [], c='grey', ls='--', lw=3.0)
    axs1[0, 0].legend([blackline, greyline], ['SBMc', 'SBMi'], loc='upper right')
    fig1.savefig('Ss_Mp_temporal_dynamics.png', dpi=600)


def distribution_plot():
    """
    Species size distributions at specific time steps from the two species competition experiment.
    :return: plot
    """

    Ss_Mp_data_path = pkg_resources.resource_filename('insidephy.examples', 'SsMp_exp.zarr')
    ds_sbmi_asyn = xr.open_zarr(Ss_Mp_data_path + '/sbmi_asyn').to_dataframe()
    ds_sbmi_asyn.spp = ds_sbmi_asyn.spp.astype('category').cat.reorder_categories(['Ss', 'Mp'], ordered=True)
    cols = ['#5494aeff', '#7cb950ff']

    fig = plt.figure(figsize=(10, 8))
    gs0 = gridspec.GridSpec(2, 1, hspace=0.20, figure=fig)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 3, wspace=0.125, hspace=0.15, subplot_spec=gs0[0])
    ax1 = fig.add_subplot(gs00[0, 0])
    ax2 = fig.add_subplot(gs00[0, 1])
    ax3 = fig.add_subplot(gs00[0, 2])
    ax4 = fig.add_subplot(gs00[1, 0])
    ax5 = fig.add_subplot(gs00[1, 1])
    ax6 = fig.add_subplot(gs00[1, 2])
    gs01 = gs0[1].subgridspec(3, 3, wspace=0.125, hspace=0.15)
    ax7 = fig.add_subplot(gs01[0, 0])
    ax8 = fig.add_subplot(gs01[0, 1])
    ax9 = fig.add_subplot(gs01[0, 2])
    ax10 = fig.add_subplot(gs01[1, 0])
    ax11 = fig.add_subplot(gs01[1, 1])
    ax12 = fig.add_subplot(gs01[1, 2])
    ax13 = fig.add_subplot(gs01[2, 0])
    ax14 = fig.add_subplot(gs01[2, 1])
    ax15 = fig.add_subplot(gs01[2, 2])

    ax1.plot(ds_sbmi_asyn.time.unique(),
             ds_sbmi_asyn.groupby(['exp', 'time']).resource.first().loc['D0.0', :] * 1e3,
             c='black', lw=3.0, clip_on=False)
    ax2.plot(ds_sbmi_asyn.time.unique(),
             ds_sbmi_asyn.groupby(['exp', 'time']).resource.first().loc['D0.25', :] * 1e3,
             c='black', lw=3.0)
    ax3.plot(ds_sbmi_asyn.time.unique(),
             ds_sbmi_asyn.groupby(['exp', 'time']).resource.first().loc['D0.5', :] * 1e3,
             c='black', lw=3.0)

    ax4.plot(ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).time.first().loc['D0.0', 'Ss', :],
             ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).rep_nind.sum().loc['D0.0', 'Ss', :],
             c=cols[0], lw=3.0)
    ax4.plot(ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).time.first().loc['D0.0', 'Mp', :],
             ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).rep_nind.sum().loc['D0.0', 'Mp', :],
             c=cols[1], lw=3.0)
    ax5.plot(ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).time.first().loc['D0.25', 'Ss', :],
             ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).rep_nind.sum().loc['D0.25', 'Ss', :],
             c=cols[0], lw=3.0)
    ax5.plot(ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).time.first().loc['D0.25', 'Mp', :],
             ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).rep_nind.sum().loc['D0.25', 'Mp', :],
             c=cols[1], lw=3.0)
    ax6.plot(ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).time.first().loc['D0.5', 'Ss', :],
             ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).rep_nind.sum().loc['D0.5', 'Ss', :],
             c=cols[0], lw=3.0)
    ax6.plot(ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).time.first().loc['D0.5', 'Mp', :],
             ds_sbmi_asyn.groupby(['exp', 'spp', 'time']).rep_nind.sum().loc['D0.5', 'Mp', :],
             c=cols[1], lw=3.0)

    axs_dist = [ax7, ax10, ax13, ax8, ax11, ax14, ax9, ax12, ax15]
    d_rates = ['D0.0', 'D0.0', 'D0.0', 'D0.25', 'D0.25', 'D0.25', 'D0.5', 'D0.5', 'D0.5']
    t_step_sbmi = [0, 10, 20, 0, 10, 20, 0, 10, 20]

    for axd, dr, ti in zip(axs_dist, d_rates, t_step_sbmi):
        sns.kdeplot(
            data=ds_sbmi_asyn.loc[
                (ds_sbmi_asyn.exp == dr) & (ds_sbmi_asyn.time == ti)],
            x='cell_size', weights='rep_nind', hue='spp', palette=cols, common_norm=True,
            legend=False, cut=5, linewidth=0.5, fill=True, ax=axd, alpha=0.9)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 0.20)
        ax.tick_params(labelbottom=False, labelleft=False)

    for ax in [ax4, ax5, ax6]:
        ax.set_yscale('log')
        ax.set_ylim(1e6, 1e12)
        ax.set_xlim(0, 20)
        ax.set_yticks([1e6, 1e8, 1e10, 1e12])
        ax.tick_params(labelleft=False)
        ax.set_xlabel('Time [days]', size=11, fontweight="bold")
        ax.set_xticklabels([r'$\bf{0}$', 5, r'$\bf{10}$', 15, r'$\bf{20}$'])

    for ax in fig.axes[6:]:
        ax.set_xscale('log')
        ax.set_xlim(1e-1, 1e2)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_ylabel('')
        ax.set_xlabel('')

    for ax in [ax7, ax10, ax13]:
        ax.set_ylabel('Density', size=11, fontweight="bold")
        ax.tick_params(labelleft=True)

    for ax in fig.axes[6:9]:
        ax.set_ylim(0, 3)

    for ax in fig.axes[9:]:
        ax.set_ylim(0, 2)

    for ax in fig.axes[12:]:
        ax.set_xlabel('Cell size [$\mu$m$^{3}$]', size=11, fontweight="bold")
        ax.tick_params(labelbottom=True)

    ax1.tick_params(labelleft=True)
    ax4.tick_params(labelleft=True)
    ax1.set_title('0% dilution rate', size=12, weight='bold')
    ax2.set_title('25% dilution rate', size=12, weight='bold')
    ax3.set_title('50% dilution rate', size=12, weight='bold')
    ax1.set_ylabel('Nutrients\n[mM N]', size=11, weight='bold')
    ax4.set_ylabel('Abundance\n[cells L$^{-1}$]', size=11, weight='bold')
    fig.text(0.90, 0.380, 't=0', size=12, fontweight="bold", rotation=270)
    fig.text(0.90, 0.260, 't=10', size=12, fontweight="bold", rotation=270)
    fig.text(0.90, 0.130, 't=20', size=12, fontweight="bold", rotation=270)

    legend_elements_ax1 = [Patch(facecolor=cols[0], edgecolor='black', label='Ss'),
                           Patch(facecolor=cols[1], edgecolor='black', label='Mp')]
    ax6.legend(handles=legend_elements_ax1, ncol=2, bbox_to_anchor=(0, 1, 1, 0), loc='upper right')
    fig.savefig('Ss_Mp_size_distribution.png', dpi=600)


if __name__ == "__main__":
    sim_run()
    temporal_dynamics_plot()
    distribution_plot()
