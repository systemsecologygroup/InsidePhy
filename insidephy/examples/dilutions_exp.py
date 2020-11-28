import numpy as np
import matplotlib.pyplot as plt
from insidephy.size_based_models.SBMi import SBMi
from insidephy.size_based_models.SBMc import SBMc
from scipy.interpolate import griddata
import matplotlib.colors as mc
from matplotlib import ticker
import matplotlib.lines as mlines
import dask
# from dask.distributed import Client

# client = Client(processes=False)
dask.config.set(scheduler='processes')


def intp_size_spec(agents_size, agents_weight, time_arr, sizerange):
    yi = agents_size[:, :, :].flatten()
    xi = time_arr.repeat(agents_size.shape[2])
    zi = agents_weight[:, :, :].flatten()
    points = np.c_[xi, yi, zi]
    Xi, Yi = np.meshgrid(time_arr, sizerange)
    Zi = griddata(points[~np.isnan(points).any(axis=1), 0:2],
                  points[~np.isnan(points).any(axis=1), 2],
                  (Xi, Yi), method='linear')
    return Xi, Yi, Zi


def pwm(size, weight, axis=1):
    return np.nansum(size * weight, axis=1) / np.nansum(weight, axis=axis)


def sim_run():
    ini_resource = 0.0002
    ini_density = [1e6]
    minsize = [1e-1]
    maxsize = [1e7]
    spp_names = ['Aa']
    nsi_spp = [500]
    numsc = [100]
    tend = 50
    dilution_rate = [0.0, 0.25, 0.50]
    time_step = 1 / (24 * 60)
    volume = 1.0

    sbmc_out = []
    sbmi_out = []
    for dr in dilution_rate:
        sbmc = dask.delayed(SBMc)(ini_resource=ini_resource, ini_density=ini_density, minsize=minsize, maxsize=maxsize,
                                  spp_names=spp_names, numsc=numsc, tend=tend, dilution_rate=dr, volume=volume)
        sbmi = dask.delayed(SBMi)(ini_resource=ini_resource, ini_density=ini_density, minsize=minsize, maxsize=maxsize,
                                  spp_names=spp_names, nsi_spp=nsi_spp, nsi_min=100, nsi_max=1000, volume=volume,
                                  time_step=time_step, time_end=tend, print_time_step=1, dilution_rate=dr)
        sbmc_out.append(sbmc)
        sbmi_out.append(sbmi)

    return dask.compute(sbmc_out, sbmi_out)

# ([sbmc_D00, sbmc_D25, sbmc_D50],) = dask.compute(sbmc_out)
# ([sbmi_D00, sbmi_D25, sbmi_D50],) = dask.compute(sbmi_out)
# ([sbmc_D00, sbmc_D25, sbmc_D50], [sbmi_D00, sbmi_D25, sbmi_D50]) = sim_run()
# """
# sbmc_D00 = SBMc(ini_resource=ini_resource, ini_density=ini_density, minsize=minsize, maxsize=maxsize,
#                 spp_names=spp_names, numsc=numsc, tend=tend, dilution_rate=dilution_rate[0], volume=volume)
# sbmi_D00 = SBMi(ini_resource=ini_resource, ini_density=ini_density, minsize=minsize, maxsize=maxsize,
#                 spp_names=spp_names, nsi_spp=nsi_spp, nsi_min=100, nsi_max=1000, volume=volume,
#                 time_step=time_step, time_end=tend, print_time_step=1, dilution_rate=dilution_rate[0])
# sbmc_D25 = SBMc(ini_resource=ini_resource, ini_density=ini_density, minsize=minsize, maxsize=maxsize,
#                 spp_names=spp_names, numsc=numsc, tend=tend, dilution_rate=dilution_rate[1], volume=volume)
# sbmi_D25 = SBMi(ini_resource=ini_resource, ini_density=ini_density, minsize=minsize, maxsize=maxsize,
#                 spp_names=spp_names, nsi_spp=nsi_spp, nsi_min=100, nsi_max=1000, volume=volume,
#                 time_step=time_step, time_end=tend, print_time_step=1, dilution_rate=dilution_rate[1])
# sbmc_D50 = SBMc(ini_resource=ini_resource, ini_density=ini_density, minsize=minsize, maxsize=maxsize,
#                 spp_names=spp_names, numsc=numsc, tend=tend, dilution_rate=dilution_rate[2], volume=volume)
# sbmi_D50 = SBMi(ini_resource=ini_resource, ini_density=ini_density, minsize=minsize, maxsize=maxsize,
#                 spp_names=spp_names, nsi_spp=nsi_spp, nsi_min=100, nsi_max=1000, volume=volume,
#                 time_step=time_step, time_end=tend, print_time_step=1, dilution_rate=dilution_rate[2])
# """


def plots():
    ([sbmc_D00, sbmc_D25, sbmc_D50], [sbmi_D00, sbmi_D25, sbmi_D50]) = sim_run()

    size_range = np.logspace(np.log10(0.1), np.log10(1e8), 1000)

    X00, Y00, Z00 = intp_size_spec(sbmi_D00.agents_size, sbmi_D00.agents_abundance, sbmi_D00.time, size_range)
    X25, Y25, Z25 = intp_size_spec(sbmi_D25.agents_size, sbmi_D25.agents_abundance, sbmi_D25.time, size_range)
    X50, Y50, Z50 = intp_size_spec(sbmi_D50.agents_size, sbmi_D50.agents_abundance, sbmi_D50.time, size_range)

    X00_biom, Y00_biom, Z00_biom = intp_size_spec(sbmi_D00.agents_size, sbmi_D00.agents_biomass, sbmi_D00.time, size_range)
    X25_biom, Y25_biom, Z25_biom = intp_size_spec(sbmi_D25.agents_size, sbmi_D25.agents_biomass, sbmi_D25.time, size_range)
    X50_biom, Y50_biom, Z50_biom = intp_size_spec(sbmi_D50.agents_size, sbmi_D50.agents_biomass, sbmi_D50.time, size_range)

    # vmax = max(sbmc_D00.abundance.max(), sbmc_D25.abundance.max(), sbmc_D50.abundance.max(), np.nanmax(Z00),
    #           np.nanmax(Z25), np.nanmax(Z50))
    # vmin = min(sbmc_D00.abundance.min(), sbmc_D25.abundance.min(), sbmc_D50.abundance.min(), np.nanmin(Z00),
    #           np.nanmin(Z25), np.nanmin(Z50))
    levs = np.logspace(np.log10(1e2), np.log10(1e7), 100)  # np.power(10, np.arange(1,9)) #np.arange(1, 1e6)
    logformat = ticker.LogFormatterMathtext()  # LogFormatter(10)
    norm = mc.BoundaryNorm(levs, 256)  # mc.LogNorm(vmin=vmin, vmax=vmax)
    ticks = np.logspace(2, 7, 6)
    levs_biom = np.logspace(np.log10(1e-4), np.log10(1e1), 100)
    norm_biom = mc.BoundaryNorm(levs_biom, 256)  # mc.LogNorm(vmin=np.log10(1e-3), vmax=np.log10(1e1))#
    ticks_biom = np.logspace(-4, 1, 6)

    cmap = 'bone_r'  # 'jet'

    fig1, axs1 = plt.subplots(6, 3, sharex='col', sharey='row', figsize=(10, 8))
    axs1[0, 0].plot(sbmc_D00.time, sbmc_D00.resource * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[0, 0].plot(sbmi_D00.time, sbmi_D00.resource * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[0, 1].plot(sbmc_D25.time, sbmc_D25.resource * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[0, 1].plot(sbmi_D25.time, sbmi_D25.resource * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[0, 2].plot(sbmc_D50.time, sbmc_D50.resource * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[0, 2].plot(sbmi_D50.time, sbmi_D50.resource * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 0].plot(sbmc_D00.time, np.sum(sbmc_D00.quota * sbmc_D00.abundance, axis=1) * 1e3, c='black', lw=3.0,
                    alpha=0.9)
    axs1[1, 0].plot(sbmi_D00.time, sbmi_D00.quota * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 1].plot(sbmc_D25.time, np.sum(sbmc_D25.quota * sbmc_D25.abundance, axis=1) * 1e3, c='black', lw=3.0,
                    alpha=0.9)
    axs1[1, 1].plot(sbmi_D25.time, sbmi_D25.quota * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[1, 2].plot(sbmc_D50.time, np.sum(sbmc_D50.quota * sbmc_D50.abundance, axis=1) * 1e3, c='black', lw=3.0,
                    alpha=0.9)
    axs1[1, 2].plot(sbmi_D50.time, sbmi_D50.quota * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 0].plot(sbmc_D00.time, np.sum(sbmc_D00.abundance, axis=1), c='black', lw=3.0, alpha=0.9)
    axs1[2, 0].plot(sbmi_D00.time, sbmi_D00.abundance, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 1].plot(sbmc_D25.time, np.sum(sbmc_D25.abundance, axis=1), c='black', lw=3.0, alpha=0.9)
    axs1[2, 1].plot(sbmi_D25.time, sbmi_D25.abundance, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[2, 2].plot(sbmc_D50.time, np.sum(sbmc_D50.abundance, axis=1), c='black', lw=3.0, alpha=0.9)
    axs1[2, 2].plot(sbmi_D50.time, sbmi_D50.abundance, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 0].plot(sbmc_D00.time, np.sum(sbmc_D00.biomass, axis=1) * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[3, 0].plot(sbmi_D00.time, sbmi_D00.biomass * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 1].plot(sbmc_D25.time, np.sum(sbmc_D25.biomass, axis=1) * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[3, 1].plot(sbmi_D25.time, sbmi_D25.biomass * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    axs1[3, 2].plot(sbmc_D50.time, np.sum(sbmc_D50.biomass, axis=1) * 1e3, c='black', lw=3.0, alpha=0.9)
    axs1[3, 2].plot(sbmi_D50.time, sbmi_D50.biomass * 1e3, c='grey', ls='--', lw=3.0, alpha=0.9)
    ax1_40 = axs1[4, 0].contourf(sbmc_D00.time, sbmc_D00.size_range, np.transpose(sbmc_D00.abundance), cmap=cmap,
                                 levels=levs, norm=norm, extend='both')
    ax1_41 = axs1[4, 1].contourf(sbmc_D25.time, sbmc_D25.size_range, np.transpose(sbmc_D25.abundance), cmap=cmap,
                                 levels=levs, norm=norm, extend='both')
    ax1_42 = axs1[4, 2].contourf(sbmc_D50.time, sbmc_D50.size_range, np.transpose(sbmc_D50.abundance), cmap=cmap,
                                 levels=levs, norm=norm, extend='both')
    ax1_50 = axs1[5, 0].contourf(X00, Y00, Z00, cmap=cmap, levels=levs, norm=norm, extend='both')
    ax1_51 = axs1[5, 1].contourf(X25, Y25, Z25, cmap=cmap, levels=levs, norm=norm, extend='both')
    ax1_52 = axs1[5, 2].contourf(X50, Y50, Z50, cmap=cmap, levels=levs, norm=norm, extend='both')
    axs1[4, 0].plot(sbmc_D00.time, pwm(sbmc_D00.size_range, sbmc_D00.abundance), c='steelblue', lw=3.0)
    axs1[4, 1].plot(sbmc_D25.time, pwm(sbmc_D25.size_range, sbmc_D25.abundance), c='steelblue', lw=3.0)
    axs1[4, 2].plot(sbmc_D50.time, pwm(sbmc_D50.size_range, sbmc_D50.abundance), c='steelblue', lw=3.0)
    axs1[5, 0].plot(sbmi_D00.time, pwm(Y00.T, Z00.T), c='steelblue', lw=3.0)
    axs1[5, 1].plot(sbmi_D25.time, pwm(Y25.T, Z25.T), c='steelblue', lw=3.0)
    axs1[5, 2].plot(sbmi_D50.time, pwm(Y50.T, Z50.T), c='steelblue', lw=3.0)
    axs1[4, 0].plot(sbmc_D00.time,#[:-1],
                    #sbmc_D00.mus,
                    np.tile(sbmc_D00.size_range, (len(sbmc_D00.time), 1))[-1, np.argmax(sbmc_D00.mus, axis=1)],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs1[4, 1].plot(sbmc_D25.time,#[:-1],
                    #sbmc_D25.mus,
                    np.tile(sbmc_D25.size_range, (len(sbmc_D25.time), 1))[-1, np.argmax(sbmc_D25.mus, axis=1)],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs1[4, 2].plot(sbmc_D50.time,#[:-1],
                    #sbmc_D50.mus,
                    np.tile(sbmc_D50.size_range, (len(sbmc_D50.time), 1))[-1, np.argmax(sbmc_D50.mus, axis=1)],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs1[5, 0].plot(sbmi_D00.time, [sbmi_D00.agents_size[:, i, j] for i, j in
                                    enumerate(np.nanargmax(sbmi_D00.agents_growth[:, :, :], axis=2).flatten())],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs1[5, 1].plot(sbmi_D25.time, [sbmi_D25.agents_size[:, i, j] for i, j in
                                    enumerate(np.nanargmax(sbmi_D25.agents_growth[:, :, :], axis=2).flatten())],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs1[5, 2].plot(sbmi_D50.time, [sbmi_D50.agents_size[:, i, j] for i, j in
                                    enumerate(np.nanargmax(sbmi_D50.agents_growth[:, :, :], axis=2).flatten())],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs1[2, 0].set_yscale('log')
    axs1[3, 0].set_yscale('log')
    axs1[4, 0].set_yscale('log')
    axs1[5, 0].set_yscale('log')
    axs1[0, 0].set_ylim(0, 0.2)
    axs1[1, 0].set_ylim(0, 1)
    axs1[2, 0].set_ylim(1e4, 1e10)
    axs1[3, 0].set_ylim(1e-3, 1e3)
    axs1[4, 0].set_ylim(1e-1, 1e8)
    axs1[5, 0].set_ylim(1e-1, 1e8)
    axs1[2, 0].set_yticks([1e4, 1e7, 1e10])
    axs1[3, 0].set_yticks([1e-3, 1e0, 1e3])
    axs1[4, 0].set_yticks([1e-1, 1e3, 1e8])
    axs1[5, 0].set_yticks([1e-1, 1e3, 1e8])
    axs1[0, 0].set_ylabel('Resource\n[mM N]', weight='bold')
    axs1[1, 0].set_ylabel('PON\n[mM N]', weight='bold')
    axs1[2, 0].set_ylabel('Abundance\n[cells L$^{-1}$]', weight='bold')
    axs1[3, 0].set_ylabel('Biomass\n[mM C]', weight='bold')
    axs1[4, 0].set_ylabel('SBMc', weight='bold')
    axs1[5, 0].set_ylabel('SBMi', weight='bold')
    axs1[5, 0].set_xlabel('Time [days]', weight='bold')
    axs1[5, 1].set_xlabel('Time [days]', weight='bold')
    axs1[5, 2].set_xlabel('Time [days]', weight='bold')
    axs1[0, 0].set_title('0% dilution rate', weight='bold')
    axs1[0, 1].set_title('25% dilution rate', weight='bold')
    axs1[0, 2].set_title('50% dilution rate', weight='bold')
    fig1.subplots_adjust(wspace=0.05)
    cbax = fig1.add_axes([0.91, 0.1, 0.01, 0.25])
    cbar = fig1.colorbar(ax1_40, cax=cbax, ticks=ticks, format=logformat)
    fig1.text(0.04, 0.16, 'Cell size [$\mu$m$^{3}$]', weight='bold', rotation=90)
    fig1.text(0.96, 0.13, 'Abundance [cells L$^{-1}$]', weight='bold', rotation=270)
    blackline = mlines.Line2D([], [], c='black', lw=3.0)
    greyline = mlines.Line2D([], [], c='grey', ls='--', lw=3.0)
    axs1[0, 0].legend([blackline, greyline], ['SBMc', 'SBMi'], loc='upper right')
    musize_line = mlines.Line2D([], [], c='steelblue', lw=3.0)
    opsize_line = mlines.Line2D([], [], c='orange', ls='-', lw=3.0)
    axs1[5, 0].legend([musize_line, opsize_line], ['Mean Size', 'Optimal Size'], loc='lower right', prop={'size': 8})
    fig1.savefig('dilutions_exp.png', dpi=600)

    fig2, axs2 = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(9, 6))
    ax2_00 = axs2[0, 0].contourf(sbmc_D00.time, sbmc_D00.size_range, np.transpose(sbmc_D00.biomass) * 1e3, cmap=cmap,
                                 levels=levs_biom, norm=norm_biom, extend='both')
    ax2_01 = axs2[0, 1].contourf(sbmc_D25.time, sbmc_D25.size_range, np.transpose(sbmc_D25.biomass) * 1e3, cmap=cmap,
                                 levels=levs_biom, norm=norm_biom, extend='both')
    ax2_02 = axs2[0, 2].contourf(sbmc_D50.time, sbmc_D50.size_range, np.transpose(sbmc_D50.biomass) * 1e3, cmap=cmap,
                                 levels=levs_biom, norm=norm_biom, extend='both')
    ax2_10 = axs2[1, 0].contourf(X00_biom, Y00_biom, Z00_biom * 1e3, cmap=cmap, levels=levs_biom, norm=norm_biom,
                                 extend='both')
    ax2_11 = axs2[1, 1].contourf(X25_biom, Y25_biom, Z25_biom * 1e3, cmap=cmap, levels=levs_biom, norm=norm_biom,
                                 extend='both')
    ax2_12 = axs2[1, 2].contourf(X50_biom, Y50_biom, Z50_biom * 1e3, cmap=cmap, levels=levs_biom, norm=norm_biom,
                                 extend='both')
    axs2[0, 0].plot(sbmc_D00.time, pwm(sbmc_D00.size_range, sbmc_D00.biomass), c='steelblue', lw=3.0)
    axs2[0, 1].plot(sbmc_D25.time, pwm(sbmc_D25.size_range, sbmc_D25.biomass), c='steelblue', lw=3.0)
    axs2[0, 2].plot(sbmc_D50.time, pwm(sbmc_D50.size_range, sbmc_D50.biomass), c='steelblue', lw=3.0)
    axs2[1, 0].plot(sbmi_D00.time, pwm(Y00_biom.T, Z00_biom.T), c='steelblue', lw=3.0)
    axs2[1, 1].plot(sbmi_D25.time, pwm(Y25_biom.T, Z25_biom.T), c='steelblue', lw=3.0)
    axs2[1, 2].plot(sbmi_D50.time, pwm(Y50_biom.T, Z50_biom.T), c='steelblue', lw=3.0)
    axs2[0, 0].plot(sbmc_D00.time,#[:-1],
                    #sbmc_D00.mus,
                    np.tile(sbmc_D00.size_range, (len(sbmc_D00.time), 1))[-1, np.argmax(sbmc_D00.mus, axis=1)],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs2[0, 1].plot(sbmc_D25.time,#[:-1],
                    #sbmc_D25.mus,
                    np.tile(sbmc_D25.size_range, (len(sbmc_D25.time), 1))[-1, np.argmax(sbmc_D25.mus, axis=1)],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs2[0, 2].plot(sbmc_D50.time,#[:-1],
                    #sbmc_D50.mus,
                    np.tile(sbmc_D50.size_range, (len(sbmc_D50.time), 1))[-1, np.argmax(sbmc_D50.mus, axis=1)],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs2[1, 0].plot(sbmi_D00.time, [sbmi_D00.agents_size[:, i, j] for i, j in
                                    enumerate(np.nanargmax(sbmi_D00.agents_growth[:, :, :], axis=2).flatten())],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs2[1, 1].plot(sbmi_D25.time, [sbmi_D25.agents_size[:, i, j] for i, j in
                                    enumerate(np.nanargmax(sbmi_D25.agents_growth[:, :, :], axis=2).flatten())],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs2[1, 2].plot(sbmi_D50.time, [sbmi_D50.agents_size[:, i, j] for i, j in
                                    enumerate(np.nanargmax(sbmi_D50.agents_growth[:, :, :], axis=2).flatten())],
                    ls='-', c='orange', lw=3.0, alpha=0.7)
    axs2[0, 0].set_yscale('log')
    axs2[1, 0].set_yscale('log')
    axs2[0, 0].set_ylim(1e-1, 1e8)
    axs2[1, 0].set_ylim(1e-1, 1e8)
    axs2[0, 0].set_ylabel('SBMc cell size [$\mu$m$^{3}$]', weight='bold')
    axs2[1, 0].set_ylabel('SBMi cell size [$\mu$m$^{3}$]', weight='bold')
    axs2[1, 0].set_xlabel('Time [days]', weight='bold')
    axs2[1, 1].set_xlabel('Time [days]', weight='bold')
    axs2[1, 2].set_xlabel('Time [days]', weight='bold')
    axs2[0, 0].set_title('0% dilution rate', weight='bold')
    axs2[0, 1].set_title('25% dilution rate', weight='bold')
    axs2[0, 2].set_title('50% dilution rate', weight='bold')
    axs2[0, 0].legend([musize_line, opsize_line], ['Mean Size', 'Optimal Size'], loc='lower right', prop={'size': 8})
    fig2.subplots_adjust(wspace=0.1, hspace=0.1)
    cbax = fig2.add_axes([0.91, 0.1, 0.01, 0.8])
    cbar = fig2.colorbar(ax2_00, cax=cbax, ticks=ticks_biom, format=logformat)
    fig2.text(0.97, 0.34, 'Biomass [mM C]', weight='bold', rotation=270)
    fig2.savefig('dilutions_exp_biomass.png', dpi=600)
