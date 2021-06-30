from insidephy.size_based_models.SBMi import SBMi
from insidephy.size_based_models.SBMc import SBMc
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
            nsi_spp=[500, 500], nsi_min=100, nsi_max=2000,
            numsc=[10, 10], tend=20, dilution_rate=[0.0, 0.25, 0.50],
            time_step=1 / 24, volume=1.0
            ):
    """
    Method to compute competition experiments between two species
    from the pool of species reported by Marañon et al (2013 Eco. Lett.)
    at three dilution rates
    :param rel_size_range: initial size variability relative to its initial size mean
    :param ss_mp_names: list with names of species as reported by Marañon et al. (2013 Eco. Lett.)
    :param nsi_spp: list with initial number of super individuals to use in SBMi model type
    :param nsi_min: minimum number of super individuals
    :param nsi_max: maximum number of super individuals
    :param numsc: list with initial number of size classes use in SBMc model type
    :param tend: Total simulation time in days
    :param dilution_rate: List with three dilution rates
    :param time_step: time steps use in SBMi model type
    :param volume: volume of flask where species compete
    :return: tuple with two list containing the results for the SBMc and the SBMi model type
    """

    cluster = LocalCluster(threads_per_worker=1)
    client = Client(cluster)
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
    sbmi_out = []
    for dr in dilution_rate:
        sbmc = dask.delayed(SBMc)(ini_resource=np.max(init_r) * 1. / 1e6, ini_density=init_d,
                                  min_size=init_min_size, max_size=init_max_size,
                                  spp_names=sp_short_names, num_sc=numsc, time_end=tend,
                                  dilution_rate=dr, volume=volume)
        sbmi = dask.delayed(SBMi)(ini_resource=np.max(init_r) * 1. / 1e6, ini_density=init_d,
                                  min_size=init_min_size, max_size=init_max_size,
                                  spp_names=sp_short_names, nsi_spp=nsi_spp, nsi_min=nsi_min, nsi_max=nsi_max,
                                  volume=volume, time_step=time_step, time_end=tend, print_time_step=1,
                                  dilution_rate=dr)
        sbmc_out.append(sbmc)
        sbmi_out.append(sbmi)

    output = dask.compute(sbmc_out, sbmi_out)
    client.close()
    cluster.close()
    return output


def save_dataset(out_file_name='SsMp_exp.nc'):
    """
    Method to save results of competition experiments for two species under three dilution rates
    using xarray and netcdf files
    :param out_file_name: name output netcdf file
    :return: ncfile
    """
    ([sbmc_D00, sbmc_D25, sbmc_D50], [sbmi_D00, sbmi_D25, sbmi_D50]) = sim_run()
    ds = xr.Dataset(
        data_vars={
            'sbmi_agents_size': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'],
                np.stack((sbmi_D00.agents_size, sbmi_D25.agents_size, sbmi_D50.agents_size))),
            'sbmi_agents_biomass': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'],
                np.stack((sbmi_D00.agents_biomass, sbmi_D25.agents_biomass, sbmi_D50.agents_biomass))),
            'sbmi_agents_abundance': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'],
                np.stack((sbmi_D00.agents_abundance, sbmi_D25.agents_abundance, sbmi_D50.agents_abundance))),
            'sbmi_agents_growth': (
                ['exp_num', 'spp_num', 'time_sbmi', 'num_agents'],
                np.stack((sbmi_D00.agents_growth, sbmi_D25.agents_growth, sbmi_D50.agents_growth))),
            'sbmi_resource':
                (['exp_num', 'time_sbmi'],
                 np.stack((sbmi_D00.resource, sbmi_D25.resource, sbmi_D50.resource))),
            'sbmi_tot_abundance':
                (['exp_num', 'time_sbmi'],
                 np.stack((sbmi_D00.abundance, sbmi_D25.abundance, sbmi_D50.abundance))),
            'sbmi_tot_biomass':
                (['exp_num', 'time_sbmi'],
                 np.stack((sbmi_D00.biomass, sbmi_D25.biomass, sbmi_D50.biomass))),
            'sbmi_tot_quota':
                (['exp_num', 'time_sbmi'],
                 np.stack((sbmi_D00.quota, sbmi_D25.quota, sbmi_D50.quota))),
            'sbmi_massbalance':
                (['exp_num', 'time_sbmi'],
                 np.stack((sbmi_D00.massbalance, sbmi_D25.massbalance, sbmi_D50.massbalance))),
            'sbmc_size':
                (['exp_num', 'idx_num_sc'],
                 np.stack((sbmc_D00.size_range, sbmc_D25.size_range, sbmc_D50.size_range))),
            'sbmc_biomass':
                (['exp_num', 'time_sbmc', 'idx_num_sc'],
                 np.stack((sbmc_D00.biomass, sbmc_D25.biomass, sbmc_D50.biomass))),
            'sbmc_abundance':
                (['exp_num', 'time_sbmc', 'idx_num_sc'],
                 np.stack((sbmc_D00.abundance, sbmc_D25.abundance, sbmc_D50.abundance))),
            'sbmc_quota':
                (['exp_num', 'time_sbmc', 'idx_num_sc'],
                 np.stack((sbmc_D00.quota, sbmc_D25.quota, sbmc_D50.quota))),
            'sbmc_growth':
                (['exp_num', 'time_sbmc', 'idx_num_sc'],
                 np.stack((sbmc_D00.mus, sbmc_D25.mus, sbmc_D50.mus))),
            'sbmc_resource':
                (['exp_num', 'time_sbmc'],
                 np.stack((sbmc_D00.resource, sbmc_D25.resource, sbmc_D50.resource)))
        },
        coords={'exp_num': np.arange(3),
                'exp_name': (['exp_num'], ['D00', 'D25', 'D50']),
                'spp_num': np.arange(2),
                'spp_name_short': (['exp_num', 'spp_num'],
                                   np.stack((sbmc_D00.spp_names, sbmc_D25.spp_names, sbmc_D50.spp_names))),
                'time_sbmi': sbmi_D00.time,
                'time_sbmc': sbmc_D00.time,
                'num_agents': np.arange(2000),
                'idx_num_sc': np.arange(20)
                },
        attrs={'title': 'Ss and Mp dilution experiments',
               'description': 'Experiments for a combination of Ss and Mp species' +
                              'Using two size-based model types ' +
                              'based on size classes (SBMc) and' +
                              'based on individuals (SBMi).',
               'simulations setup': 'relative size range: 25%' +
                                    ', dilution rates: 0%, 25% and 50%' +
                                    ', volume: 1 Litre' +
                                    ', maximum time of simulations: 20 days' +
                                    ', maximum number of size classes: 10'
                                    ', maximum number of agents: 1000',
               'time_units': 'd (days)',
               'size_units': 'um^3 (cubic micrometers)',
               'biomass_units': 'mol C / cell (mol of carbon per cell)',
               'abundance_units': 'cell / L (cells per litre)',
               'quota_units': 'mol N / mol C * cell (mol of nitrogen per mol of carbon per cell)',
               'growth_units': '1 / day (per day)',
               'resource_units': 'uM N (micro Molar of nitrogen)'
               }
    )
    ds.to_netcdf(path=out_file_name)
    return ds


def temporal_dynamics_plot():
    """
    method to plot aggregate results of competition experiments for two species under three dilution rates
    :return: plot
    """
    Ss_Mp_data_path = pkg_resources.resource_filename('insidephy.data', 'SsMp_exp.nc')
    ds = xr.load_dataset(Ss_Mp_data_path)

    fig1, axs1 = plt.subplots(4, 3, sharex='col', sharey='row', figsize=(10, 8))
    # Resources plots
    axs1[0, 0].plot(ds.time_sbmc, ds.sbmc_resource[0] * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[0, 0].plot(ds.time_sbmi, ds.sbmi_resource[0] * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[0, 1].plot(ds.time_sbmc, ds.sbmc_resource[1] * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[0, 1].plot(ds.time_sbmi, ds.sbmi_resource[1] * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[0, 2].plot(ds.time_sbmc, ds.sbmc_resource[2] * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[0, 2].plot(ds.time_sbmi, ds.sbmi_resource[2] * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    # Quota plots
    axs1[1, 0].plot(ds.time_sbmc, np.sum(ds.sbmc_quota[0] * ds.sbmc_abundance[0], axis=1) * 1e3,
                    c='black', lw=3.0, alpha=0.9)
    axs1[1, 0].plot(ds.time_sbmi, ds.sbmi_tot_quota[0] * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 1].plot(ds.time_sbmc, np.sum(ds.sbmc_quota[1] * ds.sbmc_abundance[1], axis=1) * 1e3,
                    c='black', lw=3.0, alpha=0.9)
    axs1[1, 1].plot(ds.time_sbmi, ds.sbmi_tot_quota[1] * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 2].plot(ds.time_sbmc, np.sum(ds.sbmc_quota[2] * ds.sbmc_abundance[2], axis=1) * 1e3,
                    c='black', lw=3.0, alpha=0.9)
    axs1[1, 2].plot(ds.time_sbmi, ds.sbmi_tot_quota[2] * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    # Abundance plots
    axs1[2, 0].plot(ds.time_sbmc, np.sum(ds.sbmc_abundance[0], axis=1), c='black', lw=3.0, alpha=0.9)
    axs1[2, 0].plot(ds.time_sbmi, ds.sbmi_tot_abundance[0], c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 1].plot(ds.time_sbmc, np.sum(ds.sbmc_abundance[1], axis=1), c='black', lw=3.0, alpha=0.9)
    axs1[2, 1].plot(ds.time_sbmi, ds.sbmi_tot_abundance[1], c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 2].plot(ds.time_sbmc, np.sum(ds.sbmc_abundance[2], axis=1), c='black', lw=3.0, alpha=0.9)
    axs1[2, 2].plot(ds.time_sbmi, ds.sbmi_tot_abundance[2], c='grey', ls='--', lw=3.0, alpha=0.9)
    # Biomass plots
    axs1[3, 0].plot(ds.time_sbmc, np.sum(ds.sbmc_biomass[0], axis=1) * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[3, 0].plot(ds.time_sbmi, ds.sbmi_tot_biomass[0] * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 1].plot(ds.time_sbmc, np.sum(ds.sbmc_biomass[1], axis=1) * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[3, 1].plot(ds.time_sbmi, ds.sbmi_tot_biomass[1] * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 2].plot(ds.time_sbmc, np.sum(ds.sbmc_biomass[2], axis=1) * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[3, 2].plot(ds.time_sbmi, ds.sbmi_tot_biomass[2] * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
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


def spp_weights_plot():
    """
    Method to plot biomass and abundance of the two species competition experiment.
    :return: plot
    """
    Ss_Mp_data_path = pkg_resources.resource_filename('insidephy.data', 'SsMp_exp.nc')
    ds = xr.load_dataset(Ss_Mp_data_path)
    cols = ['#5494aeff', '#7cb950ff']
    fig0, axs0 = plt.subplots(2, 3,  sharex='col', sharey='row', figsize=(8, 6))
    axs0[0, 0].plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[0, 0, :], c=cols[0], ls='--', lw=3.0)
    axs0[0, 0].plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[0, 1, :], c=cols[1], ls='--', lw=3.0)
    axs0[0, 1].plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[1, 0, :], c=cols[0], ls='--', lw=3.0)
    axs0[0, 1].plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[1, 1, :], c=cols[1], ls='--', lw=3.0)
    axs0[0, 2].plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[2, 0, :], c=cols[0], ls='--', lw=3.0)
    axs0[0, 2].plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[2, 1, :], c=cols[1], ls='--', lw=3.0)

    axs0[1, 0].plot(ds.time_sbmi, ds.sbmi_agents_biomass.sum(axis=3)[0, 0, :], c=cols[0], ls='--', lw=3.0)
    axs0[1, 0].plot(ds.time_sbmi, ds.sbmi_agents_biomass.sum(axis=3)[0, 1, :], c=cols[1], ls='--', lw=3.0)
    axs0[1, 1].plot(ds.time_sbmi, ds.sbmi_agents_biomass.sum(axis=3)[1, 0, :], c=cols[0], ls='--', lw=3.0)
    axs0[1, 1].plot(ds.time_sbmi, ds.sbmi_agents_biomass.sum(axis=3)[1, 1, :], c=cols[1], ls='--', lw=3.0)
    axs0[1, 2].plot(ds.time_sbmi, ds.sbmi_agents_biomass.sum(axis=3)[2, 0, :], c=cols[0], ls='--', lw=3.0)
    axs0[1, 2].plot(ds.time_sbmi, ds.sbmi_agents_biomass.sum(axis=3)[2, 1, :], c=cols[1], ls='--', lw=3.0)

    axs0[1, 0].set_yscale('log')
    axs0[0, 0].set_yscale('log')
    axs0[0, 0].set_title('0% dilution rate', weight='bold')
    axs0[0, 1].set_title('25% dilution rate', weight='bold')
    axs0[0, 2].set_title('50% dilution rate', weight='bold')
    axs0[0, 0].set_ylabel('Abundance\n[cells L$^{-1}$]', weight='bold')
    axs0[1, 0].set_ylabel('Biomass\n[mM C]', weight='bold')
    axs0[1, 0].set_xlabel('Time [days]', weight='bold')
    axs0[1, 1].set_xlabel('Time [days]', weight='bold')
    axs0[1, 2].set_xlabel('Time [days]', weight='bold')
    Ssline = mlines.Line2D([], [], c=cols[0], ls='--', lw=3.0)
    Mpline = mlines.Line2D([], [], c=cols[1], ls='--', lw=3.0)
    axs0[0, 0].legend([Ssline, Mpline], ['Ss', 'Mp'], loc='lower right')
    fig0.subplots_adjust(wspace=0.1, hspace=0.1)
    fig0.savefig('Ss_Mp_weights.png', dpi=600)


def distribution_plot():
    """
    Species size distributions at specific time steps from the two species competition experiment.
    :return: plot
    """

    Ss_Mp_data_path = pkg_resources.resource_filename('insidephy.data', 'SsMp_exp.nc')
    ds = xr.load_dataset(Ss_Mp_data_path)
    cols = ['#5494aeff', '#7cb950ff']
    # Reshape of data to plot size distributions of species
    dtf00 = pd.DataFrame({'cellsize': ds.sbmi_agents_size[0].values.flatten(),
                          'abundance': ds.sbmi_agents_abundance[0].values.flatten(),
                          'time': np.tile(np.repeat(ds.time_sbmi.values, ds.sbmi_agents_size[0].shape[-1]),
                                          ds.sbmi_agents_size[0].shape[0]),
                          'names': np.tile(np.repeat(ds.spp_name_short[0].values, ds.sbmi_agents_size[0].shape[-1]),
                                           ds.sbmi_agents_size[0].shape[1])})
    dtf00['logcellsize'] = dtf00['cellsize'].transform(np.log10).values
    dtf00['logabundance'] = dtf00['abundance'].transform(np.log10).values

    dtf25 = pd.DataFrame({'cellsize': ds.sbmi_agents_size[1].values.flatten(),
                          'abundance': ds.sbmi_agents_abundance[1].values.flatten(),
                          'time': np.tile(np.repeat(ds.time_sbmi.values, ds.sbmi_agents_size[1].shape[-1]),
                                          ds.sbmi_agents_size[1].shape[0]),
                          'names': np.tile(np.repeat(ds.spp_name_short[1].values, ds.sbmi_agents_size[1].shape[-1]),
                                           ds.sbmi_agents_size[1].shape[1])})
    dtf25['logcellsize'] = dtf25['cellsize'].transform(np.log10).values
    dtf25['logabundance'] = dtf25['abundance'].transform(np.log10).values

    dtf50 = pd.DataFrame({'cellsize': ds.sbmi_agents_size[2].values.flatten(),
                          'abundance': ds.sbmi_agents_abundance[2].values.flatten(),
                          'time': np.tile(np.repeat(ds.time_sbmi.values, ds.sbmi_agents_size[2].shape[-1]),
                                          ds.sbmi_agents_size[2].shape[0]),
                          'names': np.tile(np.repeat(ds.spp_name_short[2].values, ds.sbmi_agents_size[2].shape[-1]),
                                           ds.sbmi_agents_size[2].shape[1])})
    dtf50['logcellsize'] = dtf50['cellsize'].transform(np.log10).values
    dtf50['logabundance'] = dtf50['abundance'].transform(np.log10).values

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

    ax1.plot(ds.time_sbmi, ds.sbmi_resource[0, :] * 1e3, c='black', lw=3.0, clip_on=False)
    ax2.plot(ds.time_sbmi, ds.sbmi_resource[1, :] * 1e3, c='black', lw=3.0)
    ax3.plot(ds.time_sbmi, ds.sbmi_resource[2, :] * 1e3, c='black', lw=3.0)

    ax4.plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[0, 0, :], c=cols[0], lw=3.0)
    ax4.plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[0, 1, :], c=cols[1], lw=3.0)
    ax5.plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[1, 0, :], c=cols[0], lw=3.0)
    ax5.plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[1, 1, :], c=cols[1], lw=3.0)
    ax6.plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[2, 0, :], c=cols[0], lw=3.0)
    ax6.plot(ds.time_sbmi, ds.sbmi_agents_abundance.sum(axis=3)[2, 1, :], c=cols[1], lw=3.0)

    sns.kdeplot(data=dtf00[dtf00.time == 0], x='cellsize', weights='abundance', hue='names', palette=cols,
                legend=False, cut=5, fill=True, multiple='stack', linewidth=0.5, ax=ax7)
    sns.kdeplot(data=dtf00[dtf00.time == 10], x='cellsize', weights='abundance', hue='names', palette=cols,
                legend=False, cut=5, fill=True, multiple='stack', linewidth=0.5, ax=ax10)
    sns.kdeplot(data=dtf00[dtf00.time == 20], x='cellsize', weights='abundance', hue='names', palette=cols,
                legend=False, cut=5, fill=True, multiple='stack', linewidth=0.5, ax=ax13)

    sns.kdeplot(data=dtf25[dtf25.time == 0], x='cellsize', weights='abundance', hue='names', palette=cols,
                legend=False, cut=5, fill=True, multiple='stack', linewidth=0.5, ax=ax8)
    sns.kdeplot(data=dtf25[dtf25.time == 10], x='cellsize', weights='abundance', hue='names', palette=cols,
                legend=False, cut=5, fill=True, multiple='stack', linewidth=0.5, ax=ax11,)
    sns.kdeplot(data=dtf25[dtf25.time == 20], x='cellsize', weights='abundance', hue='names', palette=cols,
                legend=False, cut=5, fill=True, multiple='stack', linewidth=0.5, ax=ax14)

    sns.kdeplot(data=dtf50[dtf50.time == 0], x='cellsize', weights='abundance', hue='names', palette=cols,
                legend=False, cut=5, fill=True, multiple='stack', linewidth=0.5, ax=ax9)
    sns.kdeplot(data=dtf50[dtf50.time == 10], x='cellsize', weights='abundance', hue='names', palette=cols,
                legend=False, cut=5, fill=True, multiple='stack', linewidth=0.5, ax=ax12)
    sns.kdeplot(data=dtf50[dtf50.time == 20], x='cellsize', weights='abundance', hue='names', palette=cols,
                legend=False, cut=5, fill=True, multiple='stack', linewidth=0.5, ax=ax15)

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
        ax.set_ylim(0, 2.5)

    for ax in fig.axes[9:12]:
        ax.set_ylim(0, 1.2)

    for ax in fig.axes[12:]:
        ax.set_xlabel('Cell size [$\mu$m$^{3}$]', size=11, fontweight="bold")
        ax.tick_params(labelbottom=True)
        ax.set_ylim(0, 0.4)

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

