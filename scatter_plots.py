""" Various scatter plot routines. """

import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
from scipy.stats import pearsonr
import matplotlib
import os


matplotlib.use('Agg')

import warnings     # suppress warning of masked array/nanmean
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")


def make_scatter_lwp(X, Y, optf):
    from matplotlib.colors import LogNorm
    if isinstance(X, da.Array):
        X = X.compute()
    if isinstance(Y, da.Array):
        Y = Y.compute()

    N = X.shape[0]
    bias = np.mean(Y - X)
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    std = np.sqrt(1 / N * np.sum((Y - X - bias) ** 2))
    rms = np.sqrt(1 / N * np.sum((Y - X) ** 2))

    r, p = pearsonr(X, Y)
    ann = 'Corr    : {:.2f}\n' + \
          'RMSE    : {:.2f}\n' + \
          'bc-RMSE : {:.2f}\n' + \
          'Mean AMSR2: {:.2f}\n' + \
          'Mean Imager: {:.2f}\n' + \
          'Bias    : {:.2f}\n' + \
          'N       : {:d}'
    ann = ann.format(r, rms, std, mean_X, mean_Y, bias, N)

    xmin = np.log10(0.1)
    xmax = np.log10(400)
    ymin = np.log10(0.1)
    ymax = np.log10(400)
    x_bins = np.logspace(xmin, xmax, 34)
    y_bins = np.logspace(ymin, ymax, 34)

    dummy = np.arange(0, X.shape[0])
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(221)
    h = ax.hist2d(X, Y, bins=[x_bins, y_bins],
                  cmap=plt.get_cmap('jet'), cmin=1,
                  vmin=1, vmax=70000, norm=LogNorm())
    cbar = plt.colorbar(h[3])
    ax.plot(dummy, dummy, color='black', linestyle='--')
    ax.set_xlim(x_bins[0], x_bins[-1])
    ax.set_ylim(y_bins[0], y_bins[-1])
    ax.set_ylabel(r'CCI LWP [g m$^{-2}$]')
    ax.set_xlabel(r'AMSR2 LWP [g m$^{-2}$]')
    ax.set_title('Cloud_cci+ LWP vs. AMSR2 LWP', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax = fig.add_subplot(222)
    ax.grid(color='grey', linestyle='--')
    ax.hist(Y-X, bins=120, density=False,
            weights=(100*(np.ones(X.shape[0])/X.shape[0])), range=[-220,220], color='black')
    ax.set_xlabel(r'Difference CCI - AMSR2 [g m$^{-2}$]')
    ax.set_ylabel('Percent of data [%]')
    ax.set_title('CCI - AMSR2 LWP difference distribution', fontweight='bold')
    ax.annotate(ann, xy=(0.05, 0.65), xycoords='axes fraction')

    xmin = np.log10(0.1)
    xmax = np.log10(300)
    x_bins = np.logspace(xmin, xmax, 60)

    ax1 = fig.add_subplot(223)
    ax1.grid(color='grey', linestyle='--')
    ax1.hist(Y, bins=x_bins, density=False, weights=(100*(np.ones(X.shape[0])/X.shape[0])))
    ax1.set_title('Cloud_cci+ LWP distribution', fontweight='bold')
    ax1.set_xlabel(r'CCI LWP [g m$^{-2}$]')
    ax1.set_ylabel('Percent of data [%]')
    ax1.set_xlim(x_bins[0], x_bins[-1])
    ylim_1 = ax1.get_ylim()
    ax1.set_xscale('log')

    xmin = np.log10(0.1)
    xmax = np.log10(300)
    x_bins = np.logspace(xmin, xmax, 60)

    ax2 = fig.add_subplot(224)
    ax2.grid(color='grey', linestyle='--')
    ax2.hist(X, bins=x_bins, density=False, weights=(100*(np.ones(X.shape[0])/X.shape[0])), color='red')
    ax2.set_title('AMSR2 LWP distribution', fontweight='bold')
    ax2.set_xlabel(r'AMSR2 LWP [g m$^{-2}$]')
    ax2.set_ylabel('Percent of data [%]')
    ax2.set_xlim(x_bins[0], x_bins[-1])
    ax2.set_xscale('log')
    ylim_2 = ax2.get_ylim()

    ymax = max(ylim_1[1], ylim_2[1])
    ax1.set_ylim((0, ymax))
    ax2.set_ylim((0, ymax))

    plt.tight_layout()
    if optf is not None:
        plt.savefig(optf)
        print('SAVED: {}'.format(optf))
    else:
        plt.show()

    return ann


def make_scatter_iwp(X, Y, optf):
    from matplotlib.colors import LogNorm
    if isinstance(X, da.Array):
        X = X.compute()
    if isinstance(Y, da.Array):
        Y = Y.compute()

    N = X.shape[0]
    bias = np.mean(Y - X)
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    std = np.sqrt(1 / N * np.sum((Y - X - bias) ** 2))
    rms = np.sqrt(1 / N * np.sum((Y - X) ** 2))

    r, p = pearsonr(X, Y)
    ann = 'Corr    : {:.2f}\n' + \
          'RMSE    : {:.2f}\n' + \
          'bc-RMSE : {:.2f}\n' + \
          'Mean DARDAR: {:.2f}\n' + \
          'Mean Imager: {:.2f}\n' + \
          'Bias    : {:.2f}\n' + \
          'N       : {:d}'
    ann = ann.format(r, rms, std, mean_X, mean_Y, bias, N)

    xmin = np.log10(0.5)
    xmax = np.log10(3000)
    ymin = np.log10(0.5)
    ymax = np.log10(3000)
    x_bins = np.logspace(xmin, xmax, 30)
    y_bins = np.logspace(ymin, ymax, 30)

    dummy = np.arange(0, X.shape[0])
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(221)
    h = ax.hist2d(X, Y, bins=[x_bins, y_bins],
                  cmap=plt.get_cmap('jet'), cmin=1,
                  vmin=1, vmax=500, norm=LogNorm())
    cbar = plt.colorbar(h[3])
    ax.plot(dummy, dummy, color='black', linestyle='--')
    ax.set_xlim(x_bins[0], x_bins[-1])
    ax.set_ylim(y_bins[0], y_bins[-1])
    ax.set_ylabel(r'CCI IWP [g m$^{-2}$]')
    ax.set_xlabel(r'DARDAR IWP [g m$^{-2}$]')
    ax.set_title('Cloud_cci+ IWP vs. DARDAR IWP', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax = fig.add_subplot(222)
    ax.grid(color='grey', linestyle='--')
    ax.hist(Y-X, bins=120, density=False,
            weights=(100*(np.ones(X.shape[0])/X.shape[0])), range=[-220,220], color='black')
    ax.set_xlabel(r'Difference CCI - DARDAR [g m$^{-2}$]')
    ax.set_ylabel('Percent of data [%]')
    ax.set_title('CCI - DARDAR IWP difference distribution', fontweight='bold')
    ax.annotate(ann, xy=(0.05, 0.65), xycoords='axes fraction')

    xmin = np.log10(0.1)
    xmax = np.log10(10000)
    x_bins = np.logspace(xmin, xmax, 60)

    ax1 = fig.add_subplot(223)
    ax1.grid(color='grey', linestyle='--')
    ax1.hist(Y, bins=x_bins, density=False, weights=(100*(np.ones(X.shape[0])/X.shape[0])))
    ax1.set_title('Cloud_cci+ IWP distribution', fontweight='bold')
    ax1.set_xlabel(r'CCI IWP [g m$^{-2}$]')
    ax1.set_ylabel('Percent of data [%]')
    ax1.set_xlim(x_bins[0], x_bins[-1])
    ylim_1 = ax1.get_ylim()
    ax1.set_xscale('log')

    xmin = np.log10(0.1)
    xmax = np.log10(10000)
    x_bins = np.logspace(xmin, xmax, 60)

    ax2 = fig.add_subplot(224)
    ax2.grid(color='grey', linestyle='--')
    ax2.hist(X, bins=x_bins, density=False, weights=(100*(np.ones(X.shape[0])/X.shape[0])), color='red')
    ax2.set_title('DARDAR IWP distribution', fontweight='bold')
    ax2.set_xlabel(r'DARDAR IWP [g m$^{-2}$]')
    ax2.set_ylabel('Percent of data [%]')
    ax2.set_xlim(x_bins[0], x_bins[-1])
    ax2.set_xscale('log')
    ylim_2 = ax2.get_ylim()

    ymax = max(ylim_1[1], ylim_2[1])
    ax1.set_ylim((0, ymax))
    ax2.set_ylim((0, ymax))

    plt.tight_layout()
    if optf is not None:
        plt.savefig(optf)
        print('SAVED: {}'.format(optf))
    else:
        plt.show()

    return ann


def make_scatter_cer(X, Y, optf):
    from matplotlib.colors import LogNorm
    if isinstance(X, da.Array):
        X = X.compute()
    if isinstance(Y, da.Array):
        Y = Y.compute()

    N = X.shape[0]
    bias = np.mean(Y - X)
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    std = np.sqrt(1 / N * np.sum((Y - X - bias) ** 2))
    rms = np.sqrt(1 / N * np.sum((Y - X) ** 2))

    r, p = pearsonr(X, Y)
    ann = 'Corr    : {:.2f}\n' + \
          'RMSE    : {:.2f}\n' + \
          'bc-RMSE : {:.2f}\n' + \
          'Mean DARDAR: {:.2f}\n' + \
          'Mean Imager: {:.2f}\n' + \
          'Bias    : {:.2f}\n' + \
          'N       : {:d}'
    ann = ann.format(r, rms, std, mean_X, mean_Y, bias, N)

    xmin = np.log10(0.5)
    xmax = np.log10(600)
    ymin = np.log10(0.5)
    ymax = np.log10(600)
    x_bins = np.logspace(xmin, xmax, 34)
    y_bins = np.logspace(ymin, ymax, 34)

    dummy = np.arange(0, X.shape[0])
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(221)
    h = ax.hist2d(X, Y, bins=[x_bins, y_bins],
                  cmap=plt.get_cmap('jet'), cmin=1,
                  vmin=1, vmax=2000, norm=LogNorm())
    cbar = plt.colorbar(h[3])
    ax.plot(dummy, dummy, color='black', linestyle='--')
    ax.set_xlim(x_bins[0], x_bins[-1])
    ax.set_ylim(y_bins[0], y_bins[-1])
    ax.set_ylabel(r'CCI CER [$\mu$m]')
    ax.set_xlabel(r'DARDAR CER [$\mu$m]')
    ax.set_title('Cloud_cci+ LWP vs. DARDAR CER', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax = fig.add_subplot(222)
    ax.grid(color='grey', linestyle='--')
    ax.hist(Y-X, bins=120, density=False,
            weights=(100*(np.ones(X.shape[0])/X.shape[0])), range=[-300,300], color='black')
    ax.set_xlabel(r'Difference CCI - DARDAR [$\mu$m]')
    ax.set_ylabel('Percent of data [%]')
    ax.set_title('CCI - DARDAR CER difference distribution', fontweight='bold')
    ax.annotate(ann, xy=(0.05, 0.65), xycoords='axes fraction')

    xmin = np.log10(0.1)
    xmax = np.log10(600)
    x_bins = np.logspace(xmin, xmax, 60)

    ax1 = fig.add_subplot(223)
    ax1.grid(color='grey', linestyle='--')
    ax1.hist(Y, bins=x_bins, density=False, weights=(100*(np.ones(X.shape[0])/X.shape[0])))
    ax1.set_title('Cloud_cci+ CER distribution', fontweight='bold')
    ax1.set_xlabel(r'CCI CER [$\mu$m]')
    ax1.set_ylabel('Percent of data [%]')
    ax1.set_xlim(x_bins[0], x_bins[-1])
    ylim_1 = ax1.get_ylim()
    ax1.set_xscale('log')

    xmin = np.log10(0.1)
    xmax = np.log10(600)
    x_bins = np.logspace(xmin, xmax, 60)

    ax2 = fig.add_subplot(224)
    ax2.grid(color='grey', linestyle='--')
    ax2.hist(X, bins=x_bins, density=False, weights=(100*(np.ones(X.shape[0])/X.shape[0])), color='red')
    ax2.set_title('DARDAR CER distribution', fontweight='bold')
    ax2.set_xlabel(r'DARDAR CER [$\mu$m]')
    ax2.set_ylabel('Percent of data [%]')
    ax2.set_xlim(x_bins[0], x_bins[-1])
    ax2.set_xscale('log')
    ylim_2 = ax2.get_ylim()

    ymax = max(ylim_1[1], ylim_2[1])
    ax1.set_ylim((0, ymax))
    ax2.set_ylim((0, ymax))

    plt.tight_layout()
    if optf is not None:
        plt.savefig(optf)
        print('SAVED: {}'.format(optf))
    else:
        plt.show()

    return ann


def make_scatter_ctx(data, optf, dnt, dataset):
    """ Plot CTH/CTT scatter plots. """

    from scipy.stats import linregress
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(16, 4))
    # variable to be plotted
    vars = ['cth', 'ctt','ctp']
    # limits for plotting
    lims = {'cth': (0, 25), 'ctt': (150, 325),'ctp': (0,1100)}
    # units for plotting
    units = {'cth': 'km', 'ctt': 'm', 'ctp':'hPa'}

    for cnt, variable in enumerate(vars):
        x = data['imager_' + variable].compute()
        y = data['caliop_' + variable].compute()

        # divide CCI CTH by 1000 to convert from m to km
        if variable == 'cth': # and dataset == 'CCI':
            x /= 1000
            y /= 1000

        # dummy data for 1:1 line
        dummy = np.arange(0, lims[variable][1])

        # remove nans in both arrays
        mask = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~mask]
        y = y[~mask]

        ax = fig.add_subplot(1, 3, cnt + 1)
        h = ax.hist2d(x, y,
                      bins=(100, 100),
                      #cmap=plt.get_cmap('YlOrRd'),
                      cmap=plt.get_cmap('jet'),
                      norm=LogNorm(),
                      vmin=1,
                      vmax=1E3)

        # make linear regression
        reg = linregress(x, y)
        # plot linear regression
        ax.plot(reg[0] * dummy + reg[1], color='blue')
        # plot 1:1 line
        ax.plot(dummy, dummy, color='black')

        ax.set_xlabel('imager_{} [{}]'.format(variable, units[variable]))
        ax.set_ylabel('caliop_{} [{}]'.format(variable, units[variable]))
        ax.set_xlim(lims[variable])
        ax.set_ylim(lims[variable])
        ax.set_title(variable.upper() + ' ' + dnt, fontweight='bold')
        # write regression parameters to plot
        ax.annotate(xy=(0.05, 0.88),
                    s='r={:.2f}'.format(reg[2]),
                    xycoords='axes fraction', color='blue', fontweight='bold',
                    backgroundcolor='lightgrey')

        plt.colorbar(h[3], ax=ax)

    plt.tight_layout()
    plt.savefig(optf)
    plt.close()
    print('SAVED ', os.path.basename(optf))
