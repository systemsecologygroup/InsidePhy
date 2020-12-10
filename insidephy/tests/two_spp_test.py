from insidephy.size_based_models.SBMi import SBMi
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
import pandas as pd

ini_resource = 0.002
ini_density = [1e5, 1e10]
minsize = [1.5e7, 1.5e0]
maxsize = [2.5e7, 2.5e0]
spp_names = ['Aa', 'Bb']
nsispp = [50] * 2
nsimin = 10 * 2
nsimax = 100 * 2
tend = 20
dilution_rate = 0.0
time_step = 1 / (24*60)
volume = 1.0

twospp = SBMi(ini_resource=ini_resource, ini_density=ini_density, min_size=minsize, max_size=maxsize, spp_names=spp_names,
              dilution_rate=dilution_rate, volume=volume, nsi_spp=nsispp, nsi_min=nsimin, nsi_max=nsimax,
              time_step=time_step, time_end=tend, timeit=False)


def intp_size_spec(agents_size, agents_weight, time_arr, sizerange):
    yi = agents_size[:, :].flatten()
    xi = time_arr.repeat(agents_size.shape[-1])
    zi = agents_weight[:, :].flatten()
    points = np.c_[xi, yi, zi]
    Xi, Yi = np.meshgrid(time_arr, sizerange)
    Zi = griddata(points[~np.isnan(points).any(axis=1), 0:2],
                  points[~np.isnan(points).any(axis=1), 2],
                  (Xi, Yi), method='linear')
    return Xi, Yi, Zi


def pwm(size, weight, axis=1):
    return np.nansum(size * weight, axis=axis) / np.nansum(weight, axis=axis)


size_range = np.logspace(np.log10(0.1), np.log10(1e8), 1000)
X1, Y1, Z1 = intp_size_spec(twospp.agents_size[0], twospp.agents_abundance[0], twospp.time, size_range)
X2, Y2, Z2 = intp_size_spec(twospp.agents_size[1], twospp.agents_abundance[1], twospp.time, size_range)

fig0, axs0 = plt.subplots(1, 2, sharey='row')
axs0[0].contourf(X1, Y1, Z1)
axs0[1].contourf(X2, Y2, Z2)
axs0[0].set_yscale('log')
axs0[0].set_ylim(1e-2, 1e8)
axs0[0].plot(twospp.time, pwm(Y1.T, Z1.T), c='black', lw=3.0)
axs0[1].plot(twospp.time, pwm(Y2.T, Z2.T), c='black', lw=3.0)
axs0[0].set_title('Sp1')
axs0[1].set_title('Sp2')
axs0[0].set_xlabel('Time')
axs0[1].set_xlabel('Time')
axs0[0].set_ylabel('Cell Size')

dtf = pd.DataFrame({'cellsize': twospp.agents_size.flatten(),
                    'abundance': twospp.agents_abundance.flatten(),
                    'time': np.tile(np.repeat(twospp.time, twospp.agents_size.shape[-1]), twospp.agents_size.shape[0]),
                    'names': np.tile(np.repeat(twospp.spp_names, twospp.agents_size.shape[-1]), twospp.agents_size.shape[1])
                    })
dtf['logcellsize'] = dtf['cellsize'].transform(np.log10).values
dtf['logabundance'] = dtf['abundance'].transform(np.log10).values


def weighted_kde(x, weights, **kwargs):
    sns.kdeplot(x, weights=weights, **kwargs)


kde = sns.FacetGrid(dtf[(dtf.time == 0) | (dtf.time == 10) | (dtf.time == 20)], row='time', hue='names', aspect=2, height=2.5)
kde.map(weighted_kde, 'logcellsize', 'logabundance', log_scale=True)

fig1, axs1 = plt.subplots(3, 1, sharex='col')
sns.kdeplot(data=dtf[dtf.time == 0], x='cellsize', weights='abundance', hue='names',
            legend=False, cut=0, fill=True, multiple='stack', linewidth=0.25, ax=axs1[0], log_scale=True)
sns.kdeplot(data=dtf[dtf.time == 10], x='cellsize', weights='abundance', hue='names',
            legend=False, cut=0, fill=True, multiple='stack', linewidth=0.25, ax=axs1[1], log_scale=True)
sns.kdeplot(data=dtf[dtf.time == 20], x='cellsize', weights='abundance', hue='names',
            legend=False, cut=0, fill=True, multiple='stack', linewidth=0.25, ax=axs1[2], log_scale=True)
axs1[0].set_xscale('log')
axs1[0].set_title('time=0')
axs1[1].set_title('time=10')
axs1[2].set_title('time=20')

fig2, axs2 = plt.subplots(6, 1, sharex='col', figsize=(8, 5))
axs2[0].plot(twospp.time, twospp.resource, color='black')
axs2[1].plot(twospp.time, twospp.quota, color='black')
axs2[2].plot(twospp.time, twospp.biomass, color='black')
axs2[3].plot(twospp.time, twospp.abundance, color='black')
axs2[4].plot(twospp.time, twospp.massbalance, color='black')
axs2[5].plot(twospp.time, twospp.number_si.T)
axs2[0].set_ylabel('R')
axs2[1].set_ylabel('Q')
axs2[2].set_ylabel('B')
axs2[3].set_ylabel('A')
axs2[4].set_ylabel('MB')
axs2[5].set_ylabel('Nsi')
axs2[5].set_xlabel('Time')
