import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pkg_resources
import xarray as xr
from re import search
from cycler import cycler
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
from insidephy.size_based_models.trait_metrics import size_var_comp_sbmi
from insidephy.size_based_models.sbm import SBMc, SBMi_asyn


def figure1():
    """
    Nutrients
    """

    maranon_data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    cultures = pd.read_hdf(maranon_data_path, 'batchdtf')
    culbyspp = cultures.groupby(['Species'], observed=True)

    sbm_data_path = pkg_resources.resource_filename('insidephy.examples', 'Single_spp_exp_0percent.zarr')
    sbmc = xr.open_zarr(sbm_data_path + '/sbmc').to_dataframe()
    sbmcbyspp = sbmc.groupby('spp')
    sbmi = xr.open_zarr(sbm_data_path + '/sbmi_asyn').to_dataframe()
    sbmibysppt = sbmi.groupby(['spp', 'time'])

    markers = list(culbyspp['marker'].first().unique())
    cols = ['#5494aeff', '#7cb950ff', '#ebeb6bff', '#fdbf6fff', '#e95c47ff', '#9e0142ff']
    plt.rc('axes', prop_cycle=(cycler('color', cols)))

    fig1, axs1 = plt.subplots(4, 6, sharex='col', sharey='row', figsize=(8, 5))
    for ax1, key in zip(axs1.flatten(), culbyspp.groups.keys()):
        short_key = key[0] + key[search('_', key).span()[1]]
        ax1.scatter(culbyspp['Time_d'].get_group(key).astype('float'),
                    culbyspp['NO3_uM'].get_group(key).astype('float'),
                    marker=culbyspp['marker'].first()[key], lw=1, edgecolor=culbyspp['hexcol'].first()[key],
                    c=culbyspp['hexcol'].first()[key], s=20, alpha=0.5, zorder=2, clip_on=False)
        ax1.plot(sbmc.time.unique(),
                 sbmcbyspp.get_group(short_key).resource * 1e6,
                 c=culbyspp['hexcol'].first()[key], ls='-', zorder=3)
        ax1.plot(sbmi.time.unique(),
                 sbmibysppt.resource.first().loc[short_key, :] * 1e6,
                 c=culbyspp['hexcol'].first()[key], ls='--', zorder=2)
        ax1.set_ylim(0, 200)
        ax1.set_xlim(0, 16)
        ax1.set_xticks([0, 8, 16])
        ax1.set_xticks([4, 12], minor=True)
        ax1.set_yticks([50, 150], minor=True)
        ax1.text(11, 160, short_key, weight='bold')
        ax1.tick_params(top=True, right=True, which='both')
    axs1[2, 4].set_xticklabels([0, 8, 16])
    axs1[3, 4].axis('off')
    axs1[3, 5].axis('off')
    axs1[3, 4].plot(np.arange(7), np.arange(10, 170, 30).reshape((1, 6)) * np.ones((7, 6)), ls='-', lw=2)
    axs1[3, 4].plot(np.arange(7) + 10, np.arange(10, 170, 30).reshape((1, 6)) * np.ones((7, 6)), ls='--', lw=2)
    axs1[3, 5].scatter(1, 10, marker=markers[0], c=cols[0], s=20, lw=1, edgecolor=cols[0], alpha=0.5, zorder=3,
                       clip_on=False)
    axs1[3, 5].scatter(1, 40, marker=markers[1], c=cols[1], s=20, lw=1, edgecolor=cols[1], alpha=0.5, zorder=3,
                       clip_on=False)
    axs1[3, 5].scatter(1, 70, marker=markers[2], c=cols[2], s=20, lw=1, edgecolor=cols[2], alpha=0.5, zorder=3,
                       clip_on=False)
    axs1[3, 5].scatter(1, 100, marker=markers[3], c=cols[3], s=20, lw=1, edgecolor=cols[3], alpha=0.5, zorder=3,
                       clip_on=False)
    axs1[3, 5].scatter(1, 130, marker=markers[4], c=cols[4], s=20, lw=1, edgecolor=cols[4], alpha=0.5, zorder=3,
                       clip_on=False)
    axs1[3, 5].scatter(1, 160, marker=markers[5], c=cols[5], s=20, lw=1, edgecolor=cols[5], alpha=0.5, zorder=3,
                       clip_on=False)
    axs1[3, 5].text(5, 1, 'Cyano', weight='bold')
    axs1[3, 5].text(5, 31, 'Chloro', weight='bold')
    axs1[3, 5].text(5, 61, 'Other', weight='bold')
    axs1[3, 5].text(5, 91, 'Cocco', weight='bold')
    axs1[3, 5].text(5, 121, 'Diatom', weight='bold')
    axs1[3, 5].text(5, 151, 'Dino', weight='bold')
    fig1.text(0.05, 0.35, 'Nutrients [uM N]', rotation=90, size=12, weight='bold')
    fig1.text(0.31, 0.025, 'Time [days]', size=12, weight='bold')
    fig1.text(0.655, 0.27, 'SBMc', size=8, weight='bold')
    fig1.text(0.725, 0.27, 'SBMi', size=8, weight='bold')
    fig1.text(0.785, 0.27, 'Obs', size=8, weight='bold')
    fig1.savefig('sbm_allspp_Nutrients.png', dpi=600)


def figure2():
    """
    PON
    """

    maranon_data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    cultures = pd.read_hdf(maranon_data_path, 'batchdtf')
    culbyspp = cultures.groupby(['Species'], observed=True)

    sbm_data_path = pkg_resources.resource_filename('insidephy.examples', 'Single_spp_exp_0percent.zarr')
    sbmc = xr.open_zarr(sbm_data_path + '/sbmc').to_dataframe()
    sbmcbysppt = sbmc.groupby(['spp', 'time']).apply(lambda x: np.sum(x.quota * x.abundance) * 1e6)
    sbmi = xr.open_zarr(sbm_data_path + '/sbmi_asyn').to_dataframe()
    sbmibysppt = sbmi.groupby(['spp', 'time']).quota.sum()

    markers = list(culbyspp['marker'].first().unique())
    cols = ['#5494aeff', '#7cb950ff', '#ebeb6bff', '#fdbf6fff', '#e95c47ff', '#9e0142ff']
    plt.rc('axes', prop_cycle=(cycler('color', cols)))

    fig2, axs2 = plt.subplots(4, 6, sharex='col', sharey='row', figsize=(8, 5))
    for ax2, key in zip(axs2.flatten(), culbyspp.groups.keys()):
        short_key = key[0] + key[search('_', key).span()[1]]
        ax2.scatter(culbyspp['Time_d'].get_group(key).astype('float'),
                    culbyspp['PON_ugNmL'].get_group(key).astype('float') * 1 / 14.007 * 1e3 / 1,
                    marker=culbyspp['marker'].first()[key],
                    lw=1, edgecolor=culbyspp['hexcol'].first()[key],
                    c=culbyspp['hexcol'].first()[key], s=20, alpha=0.5, zorder=3, clip_on=False)
        ax2.plot(sbmc.time.unique(),
                 sbmcbysppt.loc[short_key, :],
                 c=culbyspp['hexcol'].first()[key], ls='-', zorder=2)
        ax2.plot(sbmi.time.unique(),
                 sbmibysppt.loc[short_key, :] * 1e6,
                 c=culbyspp['hexcol'].first()[key], ls='--', zorder=2)
        ax2.set_ylim(0, 300)
        ax2.set_xlim(0, 16)
        ax2.set_xticks([0, 8, 16])
        ax2.set_xticks([4, 12], minor=True)
        ax2.set_yticks([50, 150, 250], minor=True)
        ax2.text(11, 240, key[0] + key[search('_', key).span()[1]], weight='bold')
        ax2.tick_params(top=True, right=True, which='both')
    axs2[2, 4].set_xticklabels([0, 8, 16])
    axs2[3, 4].axis('off')
    axs2[3, 5].axis('off')
    axs2[3, 4].plot(np.arange(7), np.arange(20, 290, 45).reshape((1, 6)) * np.ones((7, 6)), ls='-', lw=2)
    axs2[3, 4].plot(np.arange(7) + 10, np.arange(20, 290, 45).reshape((1, 6)) * np.ones((7, 6)), ls='--', lw=2)
    axs2[3, 5].scatter(1, 20, marker=markers[0], c=cols[0], s=20, lw=1, edgecolor=cols[0], alpha=0.5, zorder=3,
                       clip_on=False)
    axs2[3, 5].scatter(1, 65, marker=markers[1], c=cols[1], s=20, lw=1, edgecolor=cols[1], alpha=0.5, zorder=3,
                       clip_on=False)
    axs2[3, 5].scatter(1, 110, marker=markers[2], c=cols[2], s=20, lw=1, edgecolor=cols[2], alpha=0.5, zorder=3,
                       clip_on=False)
    axs2[3, 5].scatter(1, 155, marker=markers[3], c=cols[3], s=20, lw=1, edgecolor=cols[3], alpha=0.5, zorder=3,
                       clip_on=False)
    axs2[3, 5].scatter(1, 200, marker=markers[4], c=cols[4], s=20, lw=1, edgecolor=cols[4], alpha=0.5, zorder=3,
                       clip_on=False)
    axs2[3, 5].scatter(1, 245, marker=markers[5], c=cols[5], s=20, lw=1, edgecolor=cols[5], alpha=0.5, zorder=3,
                       clip_on=False)
    axs2[3, 5].text(5, 5, 'Cyano', weight='bold')
    axs2[3, 5].text(5, 45, 'Chloro', weight='bold')
    axs2[3, 5].text(5, 90, 'Other', weight='bold')
    axs2[3, 5].text(5, 135, 'Cocco', weight='bold')
    axs2[3, 5].text(5, 180, 'Diatom', weight='bold')
    axs2[3, 5].text(5, 225, 'Dino', weight='bold')
    fig2.text(0.05, 0.4, 'PON [uM N]', rotation=90, size=12, weight='bold')
    fig2.text(0.31, 0.025, 'Time [days]', size=12, weight='bold')
    fig2.text(0.655, 0.27, 'SBMc', size=8, weight='bold')
    fig2.text(0.725, 0.27, 'SBMi', size=8, weight='bold')
    fig2.text(0.785, 0.27, 'Obs', size=8, weight='bold')
    fig2.savefig('sbm_allspp_PON.png', dpi=600)


def figure3():
    """
    Abundance
    """

    maranon_data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    cultures = pd.read_hdf(maranon_data_path, 'batchdtf')
    culbyspp = cultures.groupby(['Species'], observed=True)

    sbm_data_path = pkg_resources.resource_filename('insidephy.examples', 'Single_spp_exp_0percent.zarr')
    sbmc = xr.open_zarr(sbm_data_path + '/sbmc').to_dataframe()
    sbmcbysppt = sbmc.groupby(['spp', 'time']).apply(lambda x: np.sum(x.abundance))
    sbmi = xr.open_zarr(sbm_data_path + '/sbmi_asyn').to_dataframe()
    sbmibysppt = sbmi.groupby(['spp', 'time']).rep_nind.sum()

    markers = list(culbyspp['marker'].first().unique())
    cols = ['#5494aeff', '#7cb950ff', '#ebeb6bff', '#fdbf6fff', '#e95c47ff', '#9e0142ff']
    plt.rc('axes', prop_cycle=(cycler('color', cols)))

    ypos = np.array([2e3, 2e5, 2e7, 2e9, 2e11, 2e13])
    fig3, axs3 = plt.subplots(4, 6, sharex='col', sharey='row', figsize=(8, 5))
    for ax3, key in zip(axs3.flatten(), culbyspp.groups.keys()):
        short_key = key[0] + key[search('_', key).span()[1]]
        ax3.scatter(culbyspp['Time_d'].get_group(key).astype('float'),
                    culbyspp['Abun_cellmL'].get_group(key).astype('float') * 1e3,
                    marker=culbyspp['marker'].first()[key],
                    lw=1, edgecolor=culbyspp['hexcol'].first()[key],
                    c=culbyspp['hexcol'].first()[key], s=20, alpha=0.5, zorder=3, clip_on=False)
        ax3.plot(sbmc.time.unique(),
                 sbmcbysppt.loc[short_key, :],
                 c=culbyspp['hexcol'].first()[key], ls='-', zorder=2)
        ax3.plot(sbmi.time.unique(),
                 sbmibysppt.loc[short_key, :],
                 c=culbyspp['hexcol'].first()[key], ls='--', zorder=2)
        ax3.set_ylim(1e3, 1e15)
        ax3.set_xlim(0, 16)
        ax3.set_yscale('log')
        ax3.set_xticks([0, 8, 16])
        ax3.set_xticks([4, 12], minor=True)
        ax3.set_yticks([1e3, 1e9, 1e15])
        ax3.set_yticks([1e6, 1e12], minor=True)
        ax3.text(11, 1e13, key[0] + key[search('_', key).span()[1]], weight='bold')
        ax3.tick_params(top=True, right=True, which='both')
    axs3[3, 4].axis('off')
    axs3[3, 5].axis('off')
    axs3[3, 4].plot(np.arange(7), ypos.reshape((1, 6)) * np.ones((7, 6)), ls='-', lw=2)
    axs3[3, 4].plot(np.arange(7) + 10, ypos.reshape((1, 6)) * np.ones((7, 6)), ls='--', lw=2)
    axs3[3, 5].scatter(1, ypos[0], marker=markers[0], c=cols[0], s=20, lw=1, edgecolor=cols[0], alpha=0.5, zorder=3,
                       clip_on=False)
    axs3[3, 5].scatter(1, ypos[1], marker=markers[1], c=cols[1], s=20, lw=1, edgecolor=cols[1], alpha=0.5, zorder=3,
                       clip_on=False)
    axs3[3, 5].scatter(1, ypos[2], marker=markers[2], c=cols[2], s=20, lw=1, edgecolor=cols[2], alpha=0.5, zorder=3,
                       clip_on=False)
    axs3[3, 5].scatter(1, ypos[3], marker=markers[3], c=cols[3], s=20, lw=1, edgecolor=cols[3], alpha=0.5, zorder=3,
                       clip_on=False)
    axs3[3, 5].scatter(1, ypos[4], marker=markers[4], c=cols[4], s=20, lw=1, edgecolor=cols[4], alpha=0.5, zorder=3,
                       clip_on=False)
    axs3[3, 5].scatter(1, ypos[5], marker=markers[5], c=cols[5], s=20, lw=1, edgecolor=cols[5], alpha=0.5, zorder=3,
                       clip_on=False)
    axs3[3, 5].text(5, 1.5e3, 'Cyano', weight='bold')
    axs3[3, 5].text(5, 9.5e4, 'Chloro', weight='bold')
    axs3[3, 5].text(5, 8.5e6, 'Other', weight='bold')
    axs3[3, 5].text(5, 5.5e8, 'Cocco', weight='bold')
    axs3[3, 5].text(5, 3.5e10, 'Diatom', weight='bold')
    axs3[3, 5].text(5, 1.5e12, 'Dino', weight='bold')
    fig3.text(0.04, 0.30, 'Abundance [cells L$^{-1}$]', rotation=90, size=12, weight='bold')
    fig3.text(0.31, 0.025, 'Time [days]', size=12, weight='bold')
    fig3.text(0.655, 0.27, 'SBMc', size=8, weight='bold')
    fig3.text(0.725, 0.27, 'SBMi', size=8, weight='bold')
    fig3.text(0.785, 0.27, 'Obs', size=8, weight='bold')
    fig3.savefig('sbm_allspp_Abundance.png', dpi=600)


def figure4():
    """
    Size
    """
    maranon_data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    cultures = pd.read_hdf(maranon_data_path, 'batchdtf')
    culbyspp = cultures.groupby(['Species'], observed=True)
    sizedtf = pd.read_hdf(maranon_data_path, 'sizedtf')

    sbm_data_path = pkg_resources.resource_filename('insidephy.examples', 'Single_spp_exp_0percent.zarr')
    sbmc = xr.open_zarr(sbm_data_path + '/sbmc').to_dataframe()
    sbmcbysppt = sbmc.groupby(['spp', 'time']).apply(lambda x: np.average(x.cell_size, weights=x.abundance))
    sbmi = xr.open_zarr(sbm_data_path + '/sbmi_asyn').to_dataframe()
    sbmibysppt = sbmi.groupby(['spp', 'time']).apply(lambda x: np.average(x.cell_size, weights=x.rep_nind))

    markers = list(culbyspp['marker'].first().unique())
    cols = ['#5494aeff', '#7cb950ff', '#ebeb6bff', '#fdbf6fff', '#e95c47ff', '#9e0142ff']
    plt.rc('axes', prop_cycle=(cycler('color', cols)))

    ypos2 = np.array([2e-2, 2e0, 2e2, 2e4, 2e6, 2e8])

    fig4, axs4 = plt.subplots(4, 6, sharex='col', sharey='row', figsize=(8, 5))
    for ax4, key in zip(axs4.flatten(), culbyspp.groups.keys()):
        short_key = key[0] + key[search('_', key).span()[1]]
        ax4.plot(sbmc.time.unique(),
                 sbmcbysppt.loc[short_key, :],
                 c=culbyspp['hexcol'].first()[key], ls='-', zorder=1)
        ax4.plot(sbmi.time.unique(),
                 sbmibysppt.loc[short_key, :],
                 c=culbyspp['hexcol'].first()[key], ls='--', zorder=5)
        ax4.scatter(sizedtf.groupby('Species')['Time_d'].get_group(key).astype('float'),
                    sizedtf.groupby('Species')['Vcell'].get_group(key).astype('float'),
                    marker=sizedtf.groupby('Species')['marker'].first()[key],
                    lw=1, edgecolor=culbyspp['hexcol'].first()[key],
                    c=culbyspp['hexcol'].first()[key], s=22, alpha=0.5, zorder=10, clip_on=False)
        ax4.set_ylim(1e-2, 1e9)
        ax4.set_xlim(0, 16)
        ax4.set_yscale('log')
        ax4.set_xticks([0, 8, 16])
        ax4.set_xticks([4, 12], minor=True)
        ax4.set_yticks([1e-2, 1e3, 1e9])
        ax4.set_yticks([1e0, 1e6], minor=True)
        ax4.text(1, 1e7, key[0] + key[search('_', key).span()[1]], weight='bold')
        ax4.tick_params(top=True, right=True, which='both')
    axs4[3, 4].axis('off')
    axs4[3, 5].axis('off')
    axs4[3, 4].plot(np.arange(7), ypos2.reshape((1, 6)) * np.ones((7, 6)), ls='-', lw=2)
    axs4[3, 4].plot(np.arange(7) + 10, ypos2.reshape((1, 6)) * np.ones((7, 6)), ls='--', lw=2)
    axs4[3, 5].scatter(1, ypos2[0], marker=markers[0], c=cols[0], s=20, lw=1, edgecolor=cols[0], alpha=0.5, zorder=3,
                       clip_on=False)
    axs4[3, 5].scatter(1, ypos2[1], marker=markers[1], c=cols[1], s=20, lw=1, edgecolor=cols[1], alpha=0.5, zorder=3,
                       clip_on=False)
    axs4[3, 5].scatter(1, ypos2[2], marker=markers[2], c=cols[2], s=20, lw=1, edgecolor=cols[2], alpha=0.5, zorder=3,
                       clip_on=False)
    axs4[3, 5].scatter(1, ypos2[3], marker=markers[3], c=cols[3], s=20, lw=1, edgecolor=cols[3], alpha=0.5, zorder=3,
                       clip_on=False)
    axs4[3, 5].scatter(1, ypos2[4], marker=markers[4], c=cols[4], s=20, lw=1, edgecolor=cols[4], alpha=0.5, zorder=3,
                       clip_on=False)
    axs4[3, 5].scatter(1, ypos2[5], marker=markers[5], c=cols[5], s=20, lw=1, edgecolor=cols[5], alpha=0.5, zorder=3,
                       clip_on=False)
    axs4[3, 5].text(5, 1e-2, 'Cyano', weight='bold')
    axs4[3, 5].text(5, 1e0, 'Chloro', weight='bold')
    axs4[3, 5].text(5, 1e2, 'Other', weight='bold')
    axs4[3, 5].text(5, 1e4, 'Cocco', weight='bold')
    axs4[3, 5].text(5, 1e6, 'Diatom', weight='bold')
    axs4[3, 5].text(5, 1e8, 'Dino', weight='bold')
    fig4.text(0.04, 0.30, 'Mean cell size [$\mu$m$^{3}$]', rotation=90, size=12, weight='bold')
    fig4.text(0.31, 0.025, 'Time [days]', size=12, weight='bold')
    fig4.text(0.655, 0.28, 'SBMc', size=8, weight='bold')
    fig4.text(0.725, 0.28, 'SBMi', size=8, weight='bold')
    fig4.text(0.785, 0.28, 'Obs', size=8, weight='bold')
    fig4.savefig('sbm_allspp_CellSize.png', dpi=600)


def figure5():
    """
    Cell size distribution comparison based mean cell as initial condition.
    """
    maranon_data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    cultures = pd.read_hdf(maranon_data_path, 'batchdtf')
    culbyspp = cultures.groupby(['Species'], observed=True)
    sizedtf = pd.read_hdf(maranon_data_path, 'sizedtf')

    sbm_data_path = pkg_resources.resource_filename('insidephy.examples', 'Single_spp_exp_0percent.zarr')
    sbmc = xr.open_zarr(sbm_data_path + '/sbmc').to_dataframe()
    sbmcbyspp = sbmc.groupby(['spp'])
    sbmi = xr.open_zarr(sbm_data_path + '/sbmi_asyn').to_dataframe()
    sbmibyspp = sbmi.groupby('spp')

    cols = ['#5494aeff', '#7cb950ff', '#ebeb6bff', '#fdbf6fff', '#e95c47ff', '#9e0142ff']
    plt.rc('axes', prop_cycle=(cycler('color', cols)))

    fig5, axs5 = plt.subplots(4, 6, figsize=(10, 6))
    for ax5, key in zip(axs5.flatten(), culbyspp.groups.keys()):
        sp_name = key[0] + key[search('_', key).span()[1]]
        sns.kdeplot(data=sbmibyspp.get_group(sp_name), x='cell_size', weights='rep_nind', ax=ax5, log_scale=True,
                    fill=True, color=culbyspp['hexcol'].first()[key])
        sns.kdeplot(x=sizedtf.groupby('Species')['Vcell'].get_group(key).astype('float'), ax=ax5, fill=True,
                    color="black", common_norm=True)
        ax5.vlines(x=sbmcbyspp.get_group(sp_name).cell_size.unique(), ymin=0, ymax=100,
                   color=culbyspp['hexcol'].first()[key], ls='--')
        ax5.set_ylabel('')
        ax5.set_xlabel('')

    fig5.subplots_adjust(wspace=0.35, hspace=0.4)
    axs5[0, 0].set_ylim(0, 15)
    axs5[0, 0].set_xlim(0.01, 1)
    axs5[0, 0].text(0.015, 12.5, 'Ps', weight='bold')
    axs5[0, 1].set_ylim(0, 10)
    axs5[0, 1].set_xlim(0.01, 1)
    axs5[0, 1].text(0.015, 8.5, 'Ss', weight='bold')
    axs5[0, 2].set_ylim(0, 6)
    axs5[0, 2].set_xlim(1, 100)
    axs5[0, 2].text(3e1, 5.1, 'Os', weight='bold')
    axs5[0, 3].set_ylim(0, 15)
    axs5[0, 3].set_xlim(1, 100)
    axs5[0, 3].text(1.25, 12.5, 'Ng', weight='bold')
    axs5[0, 4].set_ylim(0, 15)
    axs5[0, 4].set_xlim(1, 100)
    axs5[0, 4].text(1.25, 12.5, 'Mp', weight='bold')
    axs5[0, 5].set_ylim(0, 15)
    axs5[0, 5].set_xlim(1e1, 1e3)
    axs5[0, 5].text(1.25e1, 12.5, 'Pl', weight='bold')
    axs5[1, 0].set_ylim(0, 20)
    axs5[1, 0].set_xlim(10, 1000)
    axs5[1, 0].text(12, 16.5, 'Cl', weight='bold')
    axs5[1, 1].set_ylim(0, 15)
    axs5[1, 1].set_xlim(10, 1000)
    axs5[1, 1].text(12, 12.5, 'Ig', weight='bold')
    axs5[1, 2].set_ylim(0, 20)
    axs5[1, 2].set_xlim(10, 1000)
    axs5[1, 2].text(12, 16.5, 'Go', weight='bold')
    axs5[1, 3].set_ylim(0, 10)
    axs5[1, 3].set_xlim(10, 1000)
    axs5[1, 3].text(12, 8.5, 'Pt', weight='bold')
    axs5[1, 4].set_ylim(0, 15)
    axs5[1, 4].set_xlim(10, 1000)
    axs5[1, 4].text(12, 12.5, 'Eh', weight='bold')
    axs5[1, 5].set_ylim(0, 15)
    axs5[1, 5].set_xlim(10, 1000)
    axs5[1, 5].text(12, 12.5, 'Sc', weight='bold')
    axs5[2, 0].set_ylim(0, 40)
    axs5[2, 0].set_xlim(100, 10000)
    axs5[2, 0].text(120, 32.5, 'Tw', weight='bold')
    axs5[2, 1].set_ylim(0, 15)
    axs5[2, 1].set_xlim(1e3, 1e5)
    axs5[2, 1].text(2e4, 12.5, 'Mn', weight='bold')
    axs5[2, 2].set_ylim(0, 15)
    axs5[2, 2].set_xlim(1e3, 1e5)
    axs5[2, 2].text(3e4, 12.5, 'Pr', weight='bold')
    axs5[2, 3].set_ylim(0, 15)
    axs5[2, 3].set_xlim(1e3, 1e5)
    axs5[2, 3].text(3e4, 12.5, 'Tr', weight='bold')
    axs5[2, 4].set_ylim(0, 10)
    axs5[2, 4].set_xlim(1e3, 1e5)
    axs5[2, 4].text(12e2, 8.5, 'Am', weight='bold')
    axs5[2, 5].set_ylim(0, 15)
    axs5[2, 5].set_xlim(1e4, 1e6)
    axs5[2, 5].text(12e3, 12.5, 'As', weight='bold')
    axs5[3, 0].set_ylim(0, 4)
    axs5[3, 0].set_xlim(1e3, 1e6)
    axs5[3, 0].text(11e2, 3.25, 'Db', weight='bold')
    axs5[3, 1].set_ylim(0, 75)
    axs5[3, 1].set_xlim(1e4, 1e6)
    axs5[3, 1].text(11e3, 62.5, 'Cr', weight='bold')
    axs5[3, 2].set_ylim(0, 4)
    axs5[3, 2].set_xlim(1e4, 1e6)
    axs5[3, 2].text(12e3, 3.25, 'At', weight='bold')
    axs5[3, 3].set_ylim(0, 6)
    axs5[3, 3].set_xlim(1e5, 1e7)
    axs5[3, 3].text(11e4, 5.0, 'Cw', weight='bold')
    axs5[3, 4].set_ylim(0, 1)
    axs5[3, 4].set_xlim(0, 1)
    axs5[3, 4].axis('off')
    axs5[3, 5].set_ylim(0, 1)
    axs5[3, 5].set_xlim(0, 1)
    axs5[3, 5].axis('off')
    axs5[3, 4].plot(np.linspace(0, 0.45, 7), np.linspace(0.05, 0.90, 6).reshape((1, 6)) * np.ones((7, 6)),
                    ls='--', lw=2)
    boxes_sbmi = [Rectangle((x, y), 0.45, 0.095) for x, y in zip(0.55 * np.ones(6), np.linspace(0.05, 0.900, 6))]
    boxes_obs = [Rectangle((x, y), 0.45, 0.095) for x, y in zip(0.0 * np.ones(6), np.linspace(0.05, 0.900, 6))]
    axs5[3, 4].add_collection(PatchCollection(boxes_sbmi, facecolors=cols, edgecolors=cols, alpha=0.50))
    axs5[3, 4].add_collection(PatchCollection(boxes_sbmi, facecolor="none", edgecolors=cols, linewidths=2))
    axs5[3, 5].add_collection(PatchCollection(boxes_obs, facecolors="black", edgecolors="black", alpha=0.50))
    axs5[3, 5].add_collection(PatchCollection(boxes_obs, facecolor="none", edgecolors="black", linewidths=2))
    fig5.text(0.86, 0.115, 'Cyano', weight='bold')
    fig5.text(0.86, 0.14, 'Chloro', weight='bold')
    fig5.text(0.86, 0.165, 'Other', weight='bold')
    fig5.text(0.86, 0.190, 'Cocco', weight='bold')
    fig5.text(0.86, 0.215, 'Diatom', weight='bold')
    fig5.text(0.86, 0.245, 'Dino', weight='bold')
    fig5.text(0.07, 0.45, 'Density', rotation=90, size=12, weight='bold')
    fig5.text(0.31, 0.025, r'Cell size [$\mu$m$^{3}$]', size=12, weight='bold')
    fig5.text(0.67, 0.08, 'SBMc', size=8, weight='bold')
    fig5.text(0.730, 0.08, 'SBMi', size=8, weight='bold')
    fig5.text(0.785, 0.08, 'Observations', size=8, weight='bold')
    fig5.savefig('sbm_allspp_CellSize_kde.png', dpi=600)


def figure6():
    """
    Effects of 0%, 25%, 50% and 75% initial cell size variability on single species experiments
    """

    def coefvar(values, weight, axis=0):
        """
        Coefficient of variation
        :param values: float, array-like, e.g. cell size
        :param weight: float, array-like, e.g. abundance or number of individuals
        :param axis: integer, axis to calculate cv, default=0
        :return: coefficient of variation
        """
        wmean = np.nansum(values * weight, axis=axis) / np.nansum(weight, axis=axis)
        wvar = np.nansum(values ** 2 * weight, axis=axis) / np.nansum(weight, axis=axis) - wmean ** 2
        if wvar < 0:
            cv = 0
        else:
            cv = np.sqrt(wvar) / wmean
        return cv

    list_files = ['Single_spp_exp_0percent.zarr', 'Single_spp_exp_25percent.zarr',
                  'Single_spp_exp_50percent.zarr', 'Single_spp_exp_75percent.zarr']

    maranon_data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
    sizedtf = pd.read_hdf(maranon_data_path, 'sizedtf')
    obsCV = pd.DataFrame({'CV': sizedtf.groupby('Species')['Vcell'].std() / sizedtf.groupby('Species')['Vcell'].mean(),
                          'Spp_name_short': [sp[0] + sp[search('_', sp).span()[1]] for sp in
                                             list(sizedtf.groupby('Species').groups.keys())]})

    cols = ['#5494aeff', '#7cb950ff', '#ebeb6bff', '#fdbf6fff', '#e95c47ff', '#9e0142ff']
    plt.rc('axes', prop_cycle=(cycler('color', cols)))

    cols_svall = ['#5494aeff', '#7cb950ff', '#ebeb6bff', '#fdbf6fff', '#e95c47ff', '#9e0142ff', 'black']
    labels_svall = ['Cyano', 'Chloro', 'Other', 'Cocco', 'Diatom', 'Dino', 'Obs']
    marker_svall = ['o', 'o', 'o', 'o', 'o', 'o', 'P']

    fig6, axs6 = plt.subplots(nrows=2, ncols=4, sharex='col', sharey='row', figsize=(10, 7))
    for fname, ncol in zip(list_files, np.arange(4)):
        sbm_data_path = pkg_resources.resource_filename('insidephy.examples', fname)
        sbmc = xr.open_zarr(sbm_data_path + '/sbmc').to_dataframe()
        sbmcbysppt = (sbmc.groupby(['spp'])
                      .apply(lambda x: coefvar(x.cell_size, x.abundance))
                      .reset_index()
                      .rename(columns={0: 'CV'})
                      )
        sbmi = xr.open_zarr(sbm_data_path + '/sbmi_asyn').to_dataframe()
        sbmibysppt = (sbmi.groupby(['spp'])
                      .apply(lambda x: coefvar(x.cell_size, x.rep_nind))
                      .reset_index()
                      .rename(columns={0: 'CV'})
                      )

        PFT_c = ['Dino', 'Dino', 'Dino', 'Cocco', 'Diatom', 'Diatom', 'Diatom', 'Cocco', 'Cocco',
                           'Cocco', 'Diatom', 'Chloro', 'Other', 'Chloro', 'Other', 'Diatom', 'Cyano', 'Dino',
                           'Diatom', 'Cyano', 'Diatom', 'Diatom']

        PFT_i = ['Dino', 'Dino', 'Dino', 'Cocco', 'Diatom', 'Diatom', 'Diatom', 'Cocco', 'Cocco',
                           'Cocco', 'Diatom', 'Chloro', 'Other', 'Chloro', 'Other', 'Diatom', 'Cyano', 'Dino',
                           'Diatom', 'Cyano', 'Diatom', 'Diatom']

        pfts = ['Cyano', 'Chloro', 'Other', 'Cocco', 'Diatom', 'Dino']
        spp_name_tag = ['Ps', 'Ss', 'Ot', 'Ng', 'Mp', 'Pl', 'Cl', 'Ig', 'Go', 'Pt', 'Eh', 'Sc',
                        'Tw', 'Mn', 'Pr', 'Tr', 'Am', 'As', 'Db', 'Cr', 'At', 'Cw']
        sbmcbysppt['Species'] = pd.Categorical(sbmcbysppt['spp'], spp_name_tag)
        sbmibysppt['Species'] = pd.Categorical(sbmibysppt['spp'], spp_name_tag)

        sns.scatterplot(x='CV', y="Species", data=sbmibysppt, hue=PFT_i,
                        hue_order=pfts, alpha=.95, ax=axs6[0, ncol], legend=False, zorder=3, clip_on=False)
        sns.scatterplot(x='CV', y="Species", data=sbmcbysppt, hue=PFT_c,
                        hue_order=pfts, alpha=.95, ax=axs6[1, ncol], legend=False, zorder=3, clip_on=False)

    for ax in axs6.ravel():
        sns.scatterplot(x='CV', y='Spp_name_short', data=obsCV, alpha=.90, marker='P', color='black', zorder=10,
                        ax=ax)
    for i in range(4):
        axs6[0, i].set_xlim(0, 1)
        axs6[0, i].set_ylabel('')
        axs6[0, i].set_xlabel('')
        axs6[1, i].set_ylabel('')
        axs6[1, i].set_xlabel('')
        axs6[1, i].set_xticks([0, 0.25, 0.5, 0.75, 1])
        axs6[1, i].set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    axs6[1, 3].legend(
        [mlines.Line2D([], [], color='w', markerfacecolor=col, marker=mak, markersize=8) for col, mak in
         zip(cols_svall, marker_svall)],
        labels_svall, prop={'size': 7, 'weight': 'bold'}, bbox_to_anchor=(1, 1.25), loc='upper left')
    fig6.subplots_adjust(wspace=0.1, hspace=0.05)
    fig6.text(0.44, 0.04, 'Cell size CV [-]', weight='bold', size=14)
    fig6.text(0.04, 0.44, 'Species', rotation=90, weight='bold', size=14)
    fig6.text(0.2, 0.89, '0%', weight='bold', size=12)
    fig6.text(0.4, 0.89, '25%', weight='bold', size=12)
    fig6.text(0.6, 0.89, '50%', weight='bold', size=12)
    fig6.text(0.8, 0.89, '75%', weight='bold', size=12)
    fig6.text(0.34, 0.92, 'Initial relative size variability [-]', weight='bold', size=14)
    fig6.text(0.92, 0.65, 'SBMi', rotation=270, weight='bold', size=14)
    fig6.text(0.92, 0.25, 'SBMc', rotation=270, weight='bold', size=14)
    fig6.savefig('sbm_svall_sizevariability.png', dpi=600)


def figure7():
    """
        Effects of 0%, 25%, 50% initial cell size variability on multiple species experiments
    """
    sbm_data_path_000 = pkg_resources.resource_filename('insidephy.examples', 'Two_spp_exp_0percent.zarr')
    sbmi_000 = xr.open_zarr(sbm_data_path_000 + '/sbmi_asyn').to_dataframe()
    dtf_000 = sbmi_000.groupby(['exp', 'time']).apply(lambda x: size_var_comp_sbmi(x))

    sbm_data_path_025 = pkg_resources.resource_filename('insidephy.examples', 'Two_spp_exp_25percent.zarr')
    sbmi_025 = xr.open_zarr(sbm_data_path_025 + '/sbmi_asyn').to_dataframe()
    dtf_025 = sbmi_025.groupby(['exp', 'time']).apply(lambda x: size_var_comp_sbmi(x))

    sbm_data_path_050 = pkg_resources.resource_filename('insidephy.examples', 'Two_spp_exp_50percent.zarr')
    sbmi_050 = xr.open_zarr(sbm_data_path_050 + '/sbmi_asyn').to_dataframe()
    dtf_050 = sbmi_050.groupby(['exp', 'time']).apply(lambda x: size_var_comp_sbmi(x))

    sbm_data_path_075 = pkg_resources.resource_filename('insidephy.examples', 'Two_spp_exp_75percent.zarr')
    sbmi_075 = xr.open_zarr(sbm_data_path_075 + '/sbmi_asyn').to_dataframe()
    dtf_075 = sbmi_075.groupby(['exp', 'time']).apply(lambda x: size_var_comp_sbmi(x))

    var_means00 = pd.DataFrame({'Intraspecific': (dtf_000.within_var / dtf_000.tot_var).groupby('exp').mean(),
                                'Interspecific': (dtf_000.between_var / dtf_000.tot_var).groupby('exp').mean(),
                                'ini_size': np.repeat('00percent', 231)})

    var_means25 = pd.DataFrame({'Intraspecific': (dtf_025.within_var / dtf_025.tot_var).groupby('exp').mean(),
                                'Interspecific': (dtf_025.between_var / dtf_025.tot_var).groupby('exp').mean(),
                                'ini_size': np.repeat('25percent', 231)})

    var_means50 = pd.DataFrame({'Intraspecific': (dtf_050.within_var / dtf_050.tot_var).groupby('exp').mean(),
                                'Interspecific': (dtf_050.between_var / dtf_050.tot_var).groupby('exp').mean(),
                                'ini_size': np.repeat('50percent', 231)})

    var_means75 = pd.DataFrame({'Intraspecific': (dtf_075.within_var / dtf_075.tot_var).groupby('exp').mean(),
                                'Interspecific': (dtf_075.between_var / dtf_075.tot_var).groupby('exp').mean(),
                                'ini_size': np.repeat('75percent', 231)})

    cmap = plt.get_cmap('bone')
    frac = 1 / 2
    fig, axs = plt.subplots(4, 1, sharex='col', figsize=(12, 6))
    var_means00.plot(kind='bar', stacked=True, color=[cmap(3 / 10), cmap(7 / 10)], ylim=(0, 1), ax=axs[0]).legend(
        loc='upper left')
    var_means25.plot(kind='bar', stacked=True, color=[cmap(3 / 10), cmap(7 / 10)], ylim=(0, 1), ax=axs[1], legend=False)
    var_means50.plot(kind='bar', stacked=True, color=[cmap(3 / 10), cmap(7 / 10)], ylim=(0, 1), ax=axs[2], legend=False)
    var_means75.plot(kind='bar', stacked=True, color=[cmap(3 / 10), cmap(7 / 10)], ylim=(0, 1), ax=axs[3], legend=False)
    axs[0].hlines(y=[frac], xmin=0, xmax=231, linestyles='dashed', colors='red')
    axs[0].text(x=215, y=0.1 + frac,
                s=str(np.round(100 * np.sum(var_means00.Intraspecific > frac) / 231, decimals=1)).zfill(1) + '%',
                c='red', weight='bold')
    axs[1].hlines(y=[frac], xmin=0, xmax=231, linestyles='dashed', colors='red')
    axs[1].text(x=215, y=0.1 + frac,
                s=str(np.round(100 * np.sum(var_means25.Intraspecific > frac) / 231, decimals=1)).zfill(1) + '%',
                c='red', weight='bold')
    axs[2].hlines(y=[frac], xmin=0, xmax=231, linestyles='dashed', colors='red')
    axs[2].text(x=215, y=0.1 + frac,
                s=str(np.round(100 * np.sum(var_means50.Intraspecific > frac) / 231, decimals=1)).zfill(1) + '%',
                c='red', weight='bold')
    axs[3].hlines(y=[frac], xmin=0, xmax=231, linestyles='dashed', colors='red')
    axs[3].text(x=215, y=0.1 + frac,
                s=str(np.round(100 * np.sum(var_means75.Intraspecific > frac) / 231, decimals=1)).zfill(1) + '%',
                c='red', weight='bold')
    axs[3].set_xticks(np.arange(0, 231, 4))
    axs[3].set_xticks(np.arange(0, 231, 1), minor=True)
    fig.text(0.075, 0.125, 'Proportion of components to total size variability ', size=12, weight='bold', rotation=90)
    fig.text(0.93, 0.25, 'Initial relative size variability [-]', size=12, weight='bold', rotation=270,
             horizontalalignment='center')
    fig.text(0.91, 0.76, '0%', size=12, weight='bold', rotation=270, horizontalalignment='center')
    fig.text(0.91, 0.56, '25%', size=12, weight='bold', rotation=270, horizontalalignment='center')
    fig.text(0.91, 0.36, '50%', size=12, weight='bold', rotation=270, horizontalalignment='center')
    fig.text(0.91, 0.16, '75%', size=12, weight='bold', rotation=270, horizontalalignment='center')
    fig.savefig('Inter_intra_size_var.png', dpi=600)

    # return var_means00, var_means25


def figure8():
    sbm_data_path_000 = pkg_resources.resource_filename('insidephy.examples', 'Two_spp_exp_0percent.zarr')
    sbmc_000 = xr.open_zarr(sbm_data_path_000 + '/sbmc').to_dataframe()
    dtf_000 = sbmc_000.groupby(['exp', 'time']).apply(lambda x: size_var_comp_sbmi(x, 'abundance'))

    sbm_data_path_025 = pkg_resources.resource_filename('insidephy.examples', 'Two_spp_exp_25percent.zarr')
    sbmc_025 = xr.open_zarr(sbm_data_path_025 + '/sbmc').to_dataframe()
    dtf_025 = sbmc_025.groupby(['exp', 'time']).apply(lambda x: size_var_comp_sbmi(x, 'abundance'))

    sbm_data_path_050 = pkg_resources.resource_filename('insidephy.examples', 'Two_spp_exp_50percent.zarr')
    sbmc_050 = xr.open_zarr(sbm_data_path_050 + '/sbmc').to_dataframe()
    dtf_050 = sbmc_050.groupby(['exp', 'time']).apply(lambda x: size_var_comp_sbmi(x, 'abundance'))

    sbm_data_path_075 = pkg_resources.resource_filename('insidephy.examples', 'Two_spp_exp_75percent.zarr')
    sbmc_075 = xr.open_zarr(sbm_data_path_075 + '/sbmc').to_dataframe()
    dtf_075 = sbmc_075.groupby(['exp', 'time']).apply(lambda x: size_var_comp_sbmi(x, 'abundance'))

    var_means00 = pd.DataFrame({'Intraspecific': (dtf_000.within_var / dtf_000.tot_var).groupby('exp').mean(),
                                'Interspecific': (dtf_000.between_var / dtf_000.tot_var).groupby('exp').mean(),
                                'ini_size': np.repeat('00percent', 231)})

    var_means25 = pd.DataFrame({'Intraspecific': (dtf_025.within_var / dtf_025.tot_var).groupby('exp').mean(),
                                'Interspecific': (dtf_025.between_var / dtf_025.tot_var).groupby('exp').mean(),
                                'ini_size': np.repeat('25percent', 231)})

    var_means50 = pd.DataFrame({'Intraspecific': (dtf_050.within_var / dtf_050.tot_var).groupby('exp').mean(),
                                'Interspecific': (dtf_050.between_var / dtf_050.tot_var).groupby('exp').mean(),
                                'ini_size': np.repeat('50percent', 231)})

    var_means75 = pd.DataFrame({'Intraspecific': (dtf_075.within_var / dtf_075.tot_var).groupby('exp').mean(),
                                'Interspecific': (dtf_075.between_var / dtf_075.tot_var).groupby('exp').mean(),
                                'ini_size': np.repeat('75percent', 231)})

    cmap = plt.get_cmap('bone')
    frac = 1 / 2
    fig, axs = plt.subplots(4, 1, sharex='col', figsize=(12, 6))
    var_means00.plot(kind='bar', stacked=True, color=[cmap(3 / 10), cmap(7 / 10)], ylim=(0, 1), ax=axs[0]).legend(
        loc='upper left')
    var_means25.plot(kind='bar', stacked=True, color=[cmap(3 / 10), cmap(7 / 10)], ylim=(0, 1), ax=axs[1], legend=False)
    var_means50.plot(kind='bar', stacked=True, color=[cmap(3 / 10), cmap(7 / 10)], ylim=(0, 1), ax=axs[2], legend=False)
    var_means75.plot(kind='bar', stacked=True, color=[cmap(3 / 10), cmap(7 / 10)], ylim=(0, 1), ax=axs[3], legend=False)
    axs[0].hlines(y=[frac], xmin=0, xmax=231, linestyles='dashed', colors='red')
    axs[0].text(x=215, y=0.1 + frac,
                s=str(np.round(100 * np.sum(var_means00.Intraspecific > frac) / 231, decimals=1)).zfill(1) + '%',
                c='red', weight='bold')
    axs[1].hlines(y=[frac], xmin=0, xmax=231, linestyles='dashed', colors='red')
    axs[1].text(x=215, y=0.1 + frac,
                s=str(np.round(100 * np.sum(var_means25.Intraspecific > frac) / 231, decimals=1)).zfill(1) + '%',
                c='red', weight='bold')
    axs[2].hlines(y=[frac], xmin=0, xmax=231, linestyles='dashed', colors='red')
    axs[2].text(x=215, y=0.1 + frac,
                s=str(np.round(100 * np.sum(var_means50.Intraspecific > frac) / 231, decimals=1)).zfill(1) + '%',
                c='red', weight='bold')
    axs[3].hlines(y=[frac], xmin=0, xmax=231, linestyles='dashed', colors='red')
    axs[3].text(x=215, y=0.1 + frac,
                s=str(np.round(100 * np.sum(var_means75.Intraspecific > frac) / 231, decimals=1)).zfill(1) + '%',
                c='red', weight='bold')
    axs[3].set_xticks(np.arange(0, 231, 4))
    axs[3].set_xticks(np.arange(0, 231, 1), minor=True)
    fig.text(0.075, 0.125, 'Proportion of components to total size variability ', size=12, weight='bold', rotation=90)
    fig.text(0.93, 0.25, 'Initial relative size variability [-]', size=12, weight='bold', rotation=270,
             horizontalalignment='center')
    fig.text(0.91, 0.76, '0%', size=12, weight='bold', rotation=270, horizontalalignment='center')
    fig.text(0.91, 0.56, '25%', size=12, weight='bold', rotation=270, horizontalalignment='center')
    fig.text(0.91, 0.36, '50%', size=12, weight='bold', rotation=270, horizontalalignment='center')
    fig.text(0.91, 0.16, '75%', size=12, weight='bold', rotation=270, horizontalalignment='center')
    fig.savefig('Inter_intra_size_var_sbmc.png', dpi=600)


def figure9(ini_resource=0.0002, ini_density=(1e4, 1e4), min_size=(1.5e1, 1.5e4), max_size=(2.5e1, 2.5e4),
             spp_names=('Aa', 'Bb'), dilution_rate=0.0, volume=1.0, nsi_spp=(500, 500), nsi_min=100,
             nsi_max=1900, num_sc=(50, 50), time_end=30, time_step=1/24, print_time_step=1):
    sbmc = SBMc(ini_resource=ini_resource, ini_density=ini_density, min_cell_size=min_size,
                max_cell_size=max_size,
                spp_names=spp_names, num_sc=num_sc, time_end=time_end,
                dilution_rate=dilution_rate, volume=volume)
    sbmi = SBMi_asyn(ini_resource=ini_resource, ini_density=ini_density, min_cell_size=min_size,
                     max_cell_size=max_size,
                     spp_names=spp_names, nsi_spp=nsi_spp, nsi_min=nsi_min, nsi_max=nsi_max,
                     volume=volume,
                     time_step=time_step, time_end=time_end, print_time_step=print_time_step,
                     dilution_rate=dilution_rate)
    cols = ['#5494aeff', '#7cb950ff']

    fig1, axs1 = plt.subplots(5, 3, sharex='col', sharey='row', figsize=(10, 8))
    # Resources plots
    axs1[0, 0].plot(sbmc.dtf.groupby('time').time.first(),
                    sbmc.dtf.groupby('time').resource.first() * 1e3, c='black', ls='--', lw=3.0, alpha=0.9)
    axs1[0, 0].plot(sbmi.dtf.time.unique(),
                    sbmi.dtf.groupby('time').resource.first() * 1e3, c='grey', lw=3.0, alpha=0.5)

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
    axs1[0, 0].legend([blackline, greyline], ['SBMc', 'SBMi'],
                      loc='lower left', prop={'size': 8})
    fig1.savefig("MREG.png", dpi=600)


def model_benchmarking(mtype,
                       ini_resource=0.0002, ini_density=(1e4, 1e4), min_size=(1.5e1, 1.5e4), max_size=(2.5e1, 2.5e4),
                       spp_names=('Aa', 'Bb'), dilution_rate=0.0, volume=1.0, nsi_spp=(100, 100), nsi_min=100,
                       nsi_max=1000, num_sc=(100, 100), time_end=10, time_step=0.1):
    if mtype == 'sbmi_asyn':
        SBMi_asyn(ini_resource=ini_resource, ini_density=ini_density, min_cell_size=min_size,
                  max_cell_size=max_size, spp_names=spp_names, dilution_rate=dilution_rate,
                  volume=volume, nsi_spp=nsi_spp, nsi_min=nsi_min,
                  nsi_max=nsi_max, time_end=time_end, time_step=time_step)
    elif mtype == 'sbmc':
        SBMc(ini_resource=ini_resource, ini_density=ini_density, min_cell_size=min_size,
             max_cell_size=max_size, spp_names=spp_names, dilution_rate=dilution_rate,
             volume=volume, num_sc=num_sc, time_end=time_end)
    else:
        raise ValueError("mtype must be a string specifying the name of the "
                         "size model type, either: sbmc, sbmi_asyn. "
                         "Instead got {!r}".format(mtype))

    """
    Benchmarking results for size-based models on Apple 2.4 GHz 8-core Intel Core i9
    Model based on size classes 
      %timeit model_benchmarking('sbmc', min_size=(2e1, 2e4), max_size=(2e1, 2e4), num_sc=(1, 1))
      18.8 ms ± 824 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
      %timeit model_benchmarking('sbmc', num_sc=(10, 10))
      21.8 ms ± 1.55 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
      %timeit model_benchmarking('sbmc', num_sc=(100, 100))
      36.1 ms ± 1.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
      %timeit model_benchmarking('sbmc', num_sc=(1000, 1000))
      237 ms ± 12.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    Model based on individuals with asynchronous updating and dt=0.1
      %timeit model_benchmarking('sbmi_asyn', nsi_min=10, nsi_max=100)
      202 ms ± 14.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
      %timeit model_benchmarking('sbmi_asyn', nsi_min=100, nsi_max=1000)
      4.12 s ± 243 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
      %timeit model_benchmarking('sbmi_asyn', nsi_min=100, nsi_max=1900)
      9.21 s ± 454 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
      %timeit model_benchmarking('sbmi_asyn', nsi_min=1000,nsi_max=10000)
      2min 9s ± 6.76 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    Model based on individuals with asynchronous updating and dt=0.01
      %timeit model_benchmarking('sbmi_asyn', nsi_min=10, nsi_max=100, time_step=0.01)
      1.47 s ± 78.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
      %timeit model_benchmarking('sbmi_asyn', nsi_min=100, nsi_max=1000, time_step=0.01)
      23.5 s ± 316 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
      %timeit model_benchmarking('sbmi_asyn', nsi_min=100, nsi_max=1900, time_step=0.01)
      59.4 s ± 506 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
      %timeit model_benchmarking('sbmi_asyn', nsi_min=1000, nsi_max=10000, time_step=0.01)
      19min 9s ± 16.1 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
