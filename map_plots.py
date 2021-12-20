import matplotlib.pyplot as plt
import matplotlib
import dask.array as da
import numpy as np
import os

matplotlib.use('Agg')

import warnings     # suppress warning of masked array/nanmean
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

def make_plot_cma_cph(scores, optf, crs, dnt, var, cosfield):
    """ Plot CMA/CPH scores. """

    fig = plt.figure(figsize=(16, 7))
    for cnt, s in enumerate(scores.keys()):
        values = scores[s]
        values[0] = da.where(scores['Nobs'][0] < 50, np.nan, values[0])
        cmap = plt.get_cmap(values[3])#.copy()
        cmap.set_bad('w')
        ax = fig.add_subplot(4, 4, cnt + 1, projection=crs)
        ims = ax.imshow(values[0],
                        transform=crs,
                        extent=crs.bounds,
                        vmin=values[1],
                        vmax=values[2],
                        cmap=cmap,
                        origin='upper',
                        interpolation='none'
                        )
        ax.coastlines(color='black')
        # mean = _weighted_spatial_average(values[0], cosfield).compute()
        # mean = '{:.2f}'.format(da.nanmean(values[0]).compute())
        mean = ''
        ax.set_title(var + ' ' + s + ' ' + dnt + ' {}'.format(mean))
        plt.colorbar(ims)
    plt.tight_layout()
    plt.savefig(optf)
    print('SAVED ', os.path.basename(optf))


def make_plot_lwp(results, crs, optf):
    """ Plot LWP stats on spatial grid. """
    import cartopy.feature as cf
    fig = plt.figure(figsize=(16, 6))

    for cnt, s in enumerate(results.keys()):
        ax = fig.add_subplot(2, 3, cnt+1, projection=crs)

        ims = ax.imshow(results[s][0],
                        transform=crs,
                        extent=crs.bounds,
                        vmin=results[s][1],
                        vmax=results[s][2],
                        cmap=plt.get_cmap(results[s][3]),
                        origin='upper',
                        interpolation='none'
                        )
        ax.add_feature(cf.OCEAN, color='grey')
        ax.add_feature(cf.LAND, color='grey')
        ax.coastlines(color='black')
        cbar = plt.colorbar(ims)
        cbar.set_label(s + r' [g m$^{-2}$]')
        ax.set_title(s, fontweight='bold')

    plt.tight_layout()
    plt.savefig(optf)
    print('SAVED ', os.path.basename(optf))


def make_plot_iwp(results, crs, optf):
    """ Plot IWP stats on spatial grid. """
    import cartopy.feature as cf
    fig = plt.figure(figsize=(16, 6))

    min_nobs = 30

    for cnt, s in enumerate(results.keys()):
        ax = fig.add_subplot(2, 3, cnt+1, projection=crs)

        if s != 'IWP Nobs':
            data = np.where(results['IWP Nobs'][0] < min_nobs, np.nan, results[s][0])
        else:
            data = results[s][0]

        ims = ax.imshow(data,
                        transform=crs,
                        extent=crs.bounds,
                        vmin=results[s][1],
                        vmax=results[s][2],
                        cmap=plt.get_cmap(results[s][3]),
                        origin='upper',
                        interpolation='none'
                        )
        ax.add_feature(cf.OCEAN, color='grey')
        ax.add_feature(cf.LAND, color='grey')
        ax.coastlines(color='black')
        cbar = plt.colorbar(ims)
        cbar.set_label(s + r' [g m$^{-2}$]')
        ax.set_title(s, fontweight='bold')

    plt.tight_layout()
    plt.savefig(optf)
    print('SAVED ', os.path.basename(optf))


def make_plot_cer(results, crs, optf):
    """ Plot LWP stats on spatial grid. """
    import cartopy.feature as cf
    fig = plt.figure(figsize=(16, 6))

    for cnt, s in enumerate(results.keys()):
        ax = fig.add_subplot(2, 3, cnt+1, projection=crs)

        ims = ax.imshow(results[s][0],
                        transform=crs,
                        extent=crs.bounds,
                        vmin=results[s][1],
                        vmax=results[s][2],
                        cmap=plt.get_cmap(results[s][3]),
                        origin='upper',
                        interpolation='none'
                        )
        ax.add_feature(cf.OCEAN, color='grey')
        ax.add_feature(cf.LAND, color='grey')
        ax.coastlines(color='black')
        cbar = plt.colorbar(ims)
        cbar.set_label(s + r' [$\mu$m]')
        ax.set_title(s, fontweight='bold')

    plt.tight_layout()
    plt.savefig(optf)
    print('SAVED ', os.path.basename(optf))


def make_plot_CTTH(scores, optf, crs, dnt, var, cosfield):
    """ Plot CTH/CTT biases. """
    fig = plt.figure(figsize=(14, 9))
    for cnt, s in enumerate(scores.keys()):
        values = scores[s]
        masked_values = np.ma.array(values[0], mask=np.isnan(values[0]))
        cmap = plt.get_cmap(values[3])#.copy()
        cmap.set_bad('grey', 1.)
        ax = fig.add_subplot(4, 3, cnt + 1, projection=crs)  # ccrs.Robinson()
        ims = ax.imshow(masked_values,
                        transform=crs,
                        extent=crs.bounds,
                        vmin=values[1],
                        vmax=values[2],
                        cmap=cmap,
                        origin='upper',
                        interpolation='none'
                        )
        ax.coastlines(color='black')
        # mean = ''
        #mean = _weighted_spatial_average(values[0], cosfield).compute()
        mean = '{:.2f}'.format(da.nanmean(values[0]).compute())
        ax.set_title(s + ' ' + dnt + ' {}'.format(mean))
        plt.colorbar(ims)
    plt.tight_layout()
    plt.savefig(optf)
    plt.close()
    print('SAVED ', os.path.basename(optf))