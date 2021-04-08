"""
Main module for plotting validation scores from atrain_match matchups on
spatial maps by calculating scores for each target gridbox. Uses bucket
resampling method.

atrain_match: https://github.com/foua-pps/atrain_match

Authors:
Daniel Philipp (DWD)
Irina Solodovnik (DWD)
"""

from pyresample import load_area
from pyresample.bucket import BucketResampler
import h5py
import os
import dask.array as da
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scores import ScoreUtils
import get_imager as gi
import get_caliop as gc

matplotlib.use('Agg')


def get_matchup_file_content(ipath, chunksize, dnt='ALL',
                             satz_lim=None, dataset='CCI'):
    """ Obtain variables from atrain_match HDF5 matchup file. """

    file = h5py.File(ipath, 'r')
    caliop = file['calipso']

    if dataset == 'CCI':
        imager = file['cci']
    elif dataset == 'CLAAS':
        imager = file['pps']
    else:
        raise Exception('Dataset {} not known!'.format(dataset))

    # get CTH and CTT
    sev_cth = da.from_array(gi.get_imager_cth(imager), chunks=chunksize)
    cal_cth = da.from_array(gc.get_caliop_cth(caliop), chunks=chunksize)
    sev_ctt = da.from_array(gi.get_imager_ctt(imager), chunks=chunksize)
    cal_ctt = da.from_array(gc.get_caliop_ctt(caliop), chunks=chunksize)
    cal_cflag = np.array(caliop['feature_classification_flags'][::, 0])

    # ctp_c = np.array(caliop['layer_top_pressure'])[:,0]
    # ctp_c = np.where(ctp_c == -9999, np.nan,ctp_c)
    # ctp_pps = np.array(imager['ctth_pressure'])
    # ctp_pps = np.where(ctp_pps==-9, np.nan, ctp_pps)
    # sev_ctp = da.from_array(ctp_pps, chunks=(chunksize))
    # cal_ctp = da.from_array(ctp_c, chunks=(chunksize))

    # get CMA, CPH, VZA, SZA, LAT and LON
    sev_cph = da.from_array(gi.get_imager_cph(imager), chunks=chunksize)
    cal_cph = da.from_array(gc.get_caliop_cph(caliop), chunks=chunksize)
    cal_cma = da.from_array(gc.get_caliop_cma(caliop), chunks=chunksize)
    sev_cma = da.from_array(gi.get_imager_cma(imager), chunks=chunksize)
    satz = da.from_array(imager['satz'], chunks=chunksize)
    sunz = da.from_array(imager['sunz'], chunks=chunksize)
    lat = da.from_array(imager['latitude'], chunks=chunksize)
    lon = da.from_array(imager['longitude'], chunks=chunksize)

    # mask satellize zenith angle
    if satz_lim is not None:
        mask = satz > satz_lim
        cal_cma = da.where(mask, np.nan, cal_cma)
        sev_cma = da.where(mask, np.nan, sev_cma)
        cal_cph = da.where(mask, np.nan, cal_cph)
        sev_cph = da.where(mask, np.nan, sev_cph)
        cal_cth = da.where(mask, np.nan, cal_cth)
        sev_cth = da.where(mask, np.nan, sev_cth)
        # cal_ctp = da.where(mask, np.nan, cal_ctt)
        # sev_ctp = da.where(mask, np.nan, sev_ctt)
    # mask all pixels except daytime
    if dnt == 'DAY':
        mask = sunz >= 80
        cal_cma = da.where(mask, np.nan, cal_cma)
        sev_cma = da.where(mask, np.nan, sev_cma)
        cal_cph = da.where(mask, np.nan, cal_cph)
        sev_cph = da.where(mask, np.nan, sev_cph)
        cal_cth = da.where(mask, np.nan, cal_cth)
        sev_cth = da.where(mask, np.nan, sev_cth)
        # cal_ctp = da.where(mask, np.nan, cal_ctt)
        # sev_ctp = da.where(mask, np.nan, sev_ctt)
    # mask all pixels except nighttime
    elif dnt == 'NIGHT':
        mask = sunz <= 95
        cal_cma = da.where(mask, np.nan, cal_cma)
        sev_cma = da.where(mask, np.nan, sev_cma)
        cal_cph = da.where(mask, np.nan, cal_cph)
        sev_cph = da.where(mask, np.nan, sev_cph)
        cal_cth = da.where(mask, np.nan, cal_cth)
        sev_cth = da.where(mask, np.nan, sev_cth)
        # cal_ctp = da.where(mask, np.nan, cal_ctt)
        # sev_ctp = da.where(mask, np.nan, sev_ctt)
    # mask all pixels except twilight
    elif dnt == 'TWILIGHT':
        mask = ~da.logical_and(sunz > 80, sunz < 95)
        cal_cma = da.where(mask, np.nan, cal_cma)
        sev_cma = da.where(mask, np.nan, sev_cma)
        cal_cph = da.where(mask, np.nan, cal_cph)
        sev_cph = da.where(mask, np.nan, sev_cph)
        cal_cth = da.where(mask, np.nan, cal_cth)
        sev_cth = da.where(mask, np.nan, sev_cth)
        # cal_ctp = da.where(mask, np.nan, cal_ctt)
        # sev_ctp = da.where(mask, np.nan, sev_ctt)
    elif dnt == 'ALL':
        pass
    else:
        raise Exception('DNT option ', dnt, ' is invalid.')

    data = {'caliop_cma': cal_cma,
            'imager_cma': sev_cma,
            'caliop_cph': cal_cph,
            'imager_cph': sev_cph,
            'satz': satz,
            'sunz': sunz,
            'caliop_cth': cal_cth,
            'imager_cth': sev_cth,
            'caliop_ctt': cal_ctt,
            'imager_ctt': sev_ctt,
            'caliop_cflag': cal_cflag}

    latlon = {'lat': lat,
              'lon': lon}

    return data, latlon


def do_cma_cph_validation(data, adef, out_size, idxs, variable):
    """ Calculate scores for CMA or CPH depending on variable arg. """

    if variable.lower() == 'cma':
        cal = data['caliop_cma']
        img = data['imager_cma']
    elif variable.lower() == 'cph':
        cal = data['caliop_cph']
        img = data['imager_cph']
    else:
        msg = 'Variable {} for CMA/CPH validation unknown. [cma, cph]'
        raise Exception(msg.format(variable))

    # pattern: CALIOP_SEVIRI
    a11 = da.logical_and(cal == 1, img == 1).astype(np.int64)
    b01 = da.logical_and(cal == 0, img == 1).astype(np.int64)
    c10 = da.logical_and(cal == 1, img == 0).astype(np.int64)
    d00 = da.logical_and(cal == 0, img == 0).astype(np.int64)

    a, _ = da.histogram(idxs, bins=out_size, range=(0, out_size),
                        weights=a11, density=False)
    b, _ = da.histogram(idxs, bins=out_size, range=(0, out_size),
                        weights=b01, density=False)
    c, _ = da.histogram(idxs, bins=out_size, range=(0, out_size),
                        weights=c10, density=False)
    d, _ = da.histogram(idxs, bins=out_size, range=(0, out_size),
                        weights=d00, density=False)

    scu = ScoreUtils(a, b, c, d)

    scores = dict()
    # [scores_on_target_grid, vmin, vmax, cmap]
    scores['Hitrate'] = [scu.hitrate().reshape(adef.shape),
                         0.5, 1, 'rainbow']
    scores['PODclr'] = [scu.pod_0().reshape(adef.shape),
                        0.5, 1, 'rainbow']
    scores['PODcld'] = [scu.pod_1().reshape(adef.shape),
                        0.5, 1, 'rainbow']
    scores['FARclr'] = [scu.far_0().reshape(adef.shape),
                        0, 1, 'rainbow']
    scores['FARcld'] = [scu.far_1().reshape(adef.shape),
                        0, 1, 'rainbow']
    scores['POFDclr'] = [scu.pofd_0().reshape(adef.shape),
                         0, 1, 'rainbow']
    scores['POFDcld'] = [scu.pofd_1().reshape(adef.shape),
                         0, 1, 'rainbow']
    scores['Heidke'] = [scu.heidke().reshape(adef.shape),
                        0, 1, 'rainbow']
    scores['Kuiper'] = [scu.kuiper().reshape(adef.shape),
                        0, 1, 'rainbow']
    scores['Bias'] = [scu.bias().reshape(adef.shape),
                      0, 1, 'bwr']
    scores['CALIOP mean'] = [scu.mean().reshape(adef.shape),
                             None, None, 'rainbow']
    scores['SEVIRI mean'] = [scu.mean().reshape(adef.shape),
                             None, None, 'rainbow']
    scores['Nobs'] = [scu.n.reshape(adef.shape),
                      None, None, 'rainbow']

    # calculate bias limits for plotting
    scores['Bias'][2] = np.nanmax(np.abs(scores['Bias'][0])) / 2
    scores['Bias'][1] = scores['Bias'][2] * (-1)
    return scores


def do_ctth_validation(data, resampler, thrs=10):
    """
    Calculate CTH and CTT bias. thrs: threshold value for filtering
    boxes with small number of obs.
    """

    # mask of detected ctth
    detected_clouds = da.logical_and(data['caliop_cma'] == 1,
                                     data['imager_cma'] == 1)
    detected_height = da.logical_and(detected_clouds,
                                     np.isfinite(data['imager_cth']))
    detected_temperature = np.logical_and(detected_clouds,
                                          np.isfinite(data['imager_ctt']))
    detected_height_mask = detected_height.astype(int)

    # calculate bias and mea for all ctth cases
    delta_h = data['imager_cth'] - data['caliop_cth']  # HEIGHT
    height_bias = np.where(detected_height, delta_h, np.nan)
    mae = np.abs(height_bias)
    delta_t = data['imager_ctt'] - data['caliop_ctt']  # TEMPERATURE
    temperature_bias = np.where(detected_temperature, delta_t, np.nan)

    # clouds levels (from calipso 'cloud type')
    low_clouds = gc.get_calipso_low_clouds(data['caliop_cflag'])
    detected_low = np.logical_and(detected_height, low_clouds)
    bias_low = np.where(detected_low, height_bias, np.nan)
    bias_temperature_low = np.where(detected_low, temperature_bias, np.nan)
    mid_clouds = gc.get_calipso_medium_clouds(data['caliop_cflag'])
    detected_mid = np.logical_and(detected_height, mid_clouds)
    bias_mid = np.where(detected_mid, height_bias, np.nan)
    high_clouds = gc.get_calipso_high_clouds(data['caliop_cflag'])
    detected_high = np.logical_and(detected_height, high_clouds)
    bias_high = np.where(detected_high, height_bias, np.nan)
    # opaque/transparent clouds (from calipso 'cloud type')
    # tp_clouds = gc.get_calipso_tp(data['caliop_cflag'])
    # detected_tp = np.logical_and(detected_height,tp_clouds)
    # bias_tp = np.where(detected_tp, height_bias, np.nan)
    # op_clouds = gc.get_calipso_op(data['caliop_cflag'])
    # detected_op = np.logical_and(detected_height,op_clouds)
    # bias_op = np.where(detected_op, height_bias, np.nan)
    # low+opaque, mid/high+transparent
    mid_high_tp_clouds = gc.get_calipso_medium_and_high_clouds_tp(
                                                data['caliop_cflag']
                                                )
    detected_mid_high_tp = np.logical_and(detected_height, mid_high_tp_clouds)
    bias_mid_high_tp = np.where(detected_mid_high_tp, height_bias, np.nan)
    low_op_clouds = gc.get_calipso_low_clouds_op(data['caliop_cflag'])
    detected_low_op = np.logical_and(detected_height, low_op_clouds)
    bias_low_op = np.where(detected_low_op, height_bias, np.nan)

    # resample and filter some data out
    # N = resampler.get_count()
    n_matched_cases = resampler.get_sum(detected_height_mask)
    sev_cth_average = resampler.get_average(data['imager_cth'])
    cal_cth_average = resampler.get_average(data['caliop_cth'])
    bias_average = resampler.get_average(height_bias)
    bias_average = np.where(n_matched_cases < thrs, np.nan, bias_average)
    mae_average = resampler.get_average(mae)
    mae_average = np.where(n_matched_cases < thrs, np.nan, mae_average)
    bias_temperature_average = resampler.get_average(temperature_bias)
    bias_temperature_average = np.where(n_matched_cases < thrs, np.nan,
                                        bias_temperature_average)

    n_matched_cases_low = resampler.get_sum(detected_low.astype(int))
    bias_low_average = resampler.get_average(bias_low)
    bias_low_average = np.where(n_matched_cases_low < thrs,
                                np.nan, bias_low_average)
    bias_temperature_low_average = resampler.get_average(bias_temperature_low)
    bias_temperature_low_average = np.where(n_matched_cases_low < thrs, np.nan,
                                            bias_temperature_low_average)
    n_matched_cases_mid = resampler.get_sum(detected_mid.astype(int))
    bias_mid_average = resampler.get_average(bias_mid)
    bias_mid_average = np.where(n_matched_cases_mid < thrs, np.nan,
                                bias_mid_average)
    n_matched_cases_high = resampler.get_sum(detected_high.astype(int))
    bias_high_average = resampler.get_average(bias_high)
    bias_high_average = np.where(n_matched_cases_high < thrs, np.nan,
                                 bias_high_average)

    # n_matched_cases_tp = resampler.get_sum(detected_tp.astype(int))
    # bias_tp_average = resampler.get_average(bias_tp)
    # bias_tp_average = np.where(n_matched_cases_tp<thrs,
    # np.nan, bias_tp_average)
    # n_matched_cases_op = resampler.get_sum(detected_op.astype(int))
    # bias_op_average = resampler.get_average(bias_op)
    # bias_op_average = np.where(n_matched_cases_op<thrs,
    # np.nan, bias_op_average)

    n_matched_cases_mid_high_tp = resampler.get_sum(
                                        detected_mid_high_tp.astype(int)
                                        )
    bias_mid_high_tp_average = resampler.get_average(bias_mid_high_tp)
    bias_mid_high_tp_average = np.where(n_matched_cases_mid_high_tp < thrs,
                                        np.nan, bias_mid_high_tp_average)
    n_matched_cases_low_op = resampler.get_sum(detected_low_op.astype(int))
    bias_low_op_average = resampler.get_average(bias_low_op)
    bias_low_op_average = np.where(n_matched_cases_low_op < thrs,
                                   np.nan, bias_low_op_average)

    # calculate scores
    scores = dict()
    # [scores_on_target_grid, vmin, vmax, cmap]
    scores['Bias CTH'] = [bias_average, -4000, 4000, 'bwr']
    scores['MAE CTH'] = [mae_average, 0, 2500, 'Reds']

    scores['Bias low'] = [bias_low_average, -2000, 2000, 'bwr']
    scores['Bias middle'] = [bias_mid_average, -2000, 2000, 'bwr']
    scores['Bias high'] = [bias_high_average, -6000, 6000, 'bwr']

    # scores['Bias opaque'] = [bias_op_average, -4000, 4000, 'bwr']
    # scores['Bias transparent'] = [bias_tp_average, -4000, 4000, 'bwr']
    scores['Bias low opaque'] = [bias_low_op_average, -2000, 2000, 'bwr']
    scores['Bias mid+high transparent'] = [bias_mid_high_tp_average,
                                           -6000, 6000, 'bwr']

    scores['Bias temperature'] = [bias_temperature_average, -30, 30, 'bwr']
    scores['Bias temperature low'] = [bias_temperature_low_average,
                                      -10, 10, 'bwr']

    scores['CALIOP CTH mean'] = [cal_cth_average, 1000, 14000, 'rainbow']
    scores['SEVIRI CTH mean'] = [sev_cth_average, 1000, 14000, 'rainbow']
    scores['Num_detected_height'] = [n_matched_cases, None, None, 'rainbow']

    # addit_scores = do_ctp_validation(data, adef, out_size, idxs)
    # scores.update(addit_scores)

    # scores['N_matched_cases_low'] = [N_matched_cases_low,
    # None, None, 'rainbow']
    # scores['N_matched_cases_middle'] = [N_matched_cases_mid,
    # None, None, 'rainbow']
    # scores['N_matched_cases_high'] = [N_matched_cases_high,
    # None, None, 'rainbow']
    return scores


def do_ctp_validation(data, adef, out_size, idxs):
    """ Calculate CTP validation (NOT AVAILABLE YET). """

    # detected ctth mask
    detected_clouds = da.logical_and(data['caliop_cma'] == 1,
                                     data['imager_cma'] == 1)
    detected_height = da.logical_and(detected_clouds,
                                     np.isfinite(data['imager_cth']))
    # find pps low and caliop low
    low_clouds_c = get_calipso_low_clouds(data['caliop_cflag'])
    detected_low_c = np.logical_and(detected_height, low_clouds_c)
    low_clouds_pps = da.where(data['imager_ctp'] > 680., 1, 0)
    detected_low_pps = da.logical_and(detected_height, low_clouds_pps)

    # pattern: CALIOP_SEVIRI
    cld_cld_a = da.logical_and(detected_low_c == 1, detected_low_pps == 1)
    clr_cld_b = da.logical_and(detected_low_c == 0, detected_low_pps == 1)
    cld_clr_c = da.logical_and(detected_low_c == 1, detected_low_pps == 0)
    clr_clr_d = da.logical_and(detected_low_c == 0, detected_low_pps == 0)

    cld_cld_a = cld_cld_a.astype(np.int64)
    clr_cld_b = clr_cld_b.astype(np.int64)
    cld_clr_c = cld_clr_c.astype(np.int64)
    clr_clr_d = clr_clr_d.astype(np.int64)

    a, _ = da.histogram(idxs, bins=out_size, range=(0, out_size),
                        weights=cld_cld_a, density=False)
    b, _ = da.histogram(idxs, bins=out_size, range=(0, out_size),
                        weights=clr_cld_b, density=False)
    c, _ = da.histogram(idxs, bins=out_size, range=(0, out_size),
                        weights=cld_clr_c, density=False)
    d, _ = da.histogram(idxs, bins=out_size, range=(0, out_size),
                        weights=clr_clr_d, density=False)

    # n = a + b + c + d
    # n2d = N.reshape(adef.shape)

    # scores = [hitrate(a, d, n).reshape(adef.shape),
    # 0.7, 1, 'rainbow'] # hitrate low PPS
    pod_low = a / (a + c)
    far_low = c / (a + c)
    scores = dict()
    scores['POD low clouds'] = [pod_low.reshape(adef.shape), 0.2, 1, 'rainbow']
    scores['FAR low clouds'] = [far_low.reshape(adef.shape), 0.2, 1, 'rainbow']

    return scores


def get_cosfield(lat):
    """ Calculate 2D cos(lat) field for weighted averaging on regular grid."""

    latcos = np.abs(np.cos(lat * np.pi / 180))
    cosfield = da.from_array(latcos, chunks=(1000, 1000))  # [mask]
    return cosfield


def weighted_spatial_average(data, cosfield):
    """ Calculate weighted spatial average. """

    if isinstance(data, xr.DataArray):
        data = data.data
    if isinstance(data, np.ndarray):
        data = da.from_array(data, chunks=(1000, 1000))
    return da.nansum(data * cosfield) / da.nansum(cosfield)


def make_plot(scores, optf, crs, dnt, var, cosfield):
    """ Plot CMA/CPH scores. """

    fig = plt.figure(figsize=(16, 7))
    for cnt, s in enumerate(scores.keys()):
        values = scores[s]
        values[0] = da.where(scores['Nobs'][0] < 50, np.nan, values[0])
        ax = fig.add_subplot(4, 4, cnt + 1, projection=crs)
        ims = ax.imshow(values[0],
                        transform=crs,
                        extent=crs.bounds,
                        vmin=values[1],
                        vmax=values[2],
                        cmap=plt.get_cmap(values[3]),
                        origin='upper',
                        interpolation='none'
                        )
        ax.coastlines(color='black')
        # mean = weighted_spatial_average(values[0], cosfield).compute()
        # mean = '{:.2f}'.format(da.nanmean(values[0]).compute())
        mean = ''
        ax.set_title(var + ' ' + s + ' ' + dnt + ' {}'.format(mean))
        plt.colorbar(ims)
    plt.tight_layout()
    plt.savefig(optf)
    print('SAVED ', os.path.basename(optf))


def make_plot_CTTH(scores, optf, crs, dnt, var, cosfield):
    """ Plot CTH/CTT biases. """
    fig = plt.figure(figsize=(16, 12))
    for cnt, s in enumerate(scores.keys()):
        values = scores[s]
        masked_values = np.ma.array(values[0], mask=np.isnan(values[0]))
        cmap = plt.get_cmap(values[3])
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
        mean = weighted_spatial_average(values[0], cosfield).compute()
        mean = '{:.2f}'.format(da.nanmean(values[0]).compute())
        ax.set_title(var + ' ' + s + ' ' + dnt + ' {}'.format(mean))
        plt.colorbar(ims)
    plt.tight_layout()
    plt.savefig(optf)
    plt.close()
    print('SAVED ', os.path.basename(optf))


def make_scatter(data, optf, dnt, dataset):
    """ Plot CTH/CTT scatter plots. """

    from scipy.stats import linregress
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(12, 4))
    # variable to be plotted
    vars = ['cth', 'ctt']
    # limits for plotting
    lims = {'cth': (0, 25), 'ctt': (150, 325)}
    # units for plotting
    units = {'cth': 'km', 'ctt': 'm'}

    for cnt, variable in enumerate(vars):
        x = data['imager_' + variable].compute()
        y = data['caliop_' + variable].compute()

        # divide CCI CTH by 1000 to convert from m to km
        if variable == 'cth' and dataset == 'CCI':
            x /= 1000
            y /= 1000

        # dummy data for 1:1 line
        dummy = np.arange(0, lims[variable][1])

        # remove nans in both arrays
        mask = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~mask]
        y = y[~mask]

        ax = fig.add_subplot(1, 2, cnt + 1)
        h = ax.hist2d(x, y,
                      bins=(100, 100),
                      cmap=plt.get_cmap('YlOrRd'),
                      norm=LogNorm())

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
        ax.annotate(xy=(0.05, 0.9),
                    s='r={:.2f}\nr**2={:.2f}'.format(reg[2], reg[2] * reg[2]),
                    xycoords='axes fraction', color='blue', fontweight='bold',
                    backgroundcolor='lightgrey')

        plt.colorbar(h[3], ax=ax)

    plt.tight_layout()
    plt.savefig(optf)
    plt.close()
    print('SAVED ', os.path.basename(optf))


def run(ipath, ifile, opath, dnts, satzs,
        year, month, dataset, chunksize=100000,
        plot_area='pc_world'):
    """
    Main function to be called in external script to run atrain_plot.

    ipath (str):     Path to HDF5 atrain_match matchup file.
    ifile (str):     HDF5 atrain_match matchup file filename.
    opath (str):     Path where figures should be saved.
    dnts (list):     List of illumination scenarios to be processed.
                     Available: ALL, DAY, NIGHT, TWILIGHT
    satzs (list):    List of satellite zenith limitations to be processed. Use
                     [None] if no limitation required.
    year (str):      String of year.
    month (str):     String of month.
    dataset (str):   Dataset validated: Available: CCI, CLAAS3.
    chunksize (int): Size of data chunks to reduce memory usage.
    plot_area (str): Name of area definition in areas.yaml file to be used.
    """

    # if dnts is single string convert to list
    if isinstance(dnts, str):
        dnts = [dnts]

    # if satzs is single string/int/float convert to list
    if isinstance(satzs, str) or isinstance(satzs, int) or isinstance(satzs, float):
        satzs = [satzs]

    if dataset not in ['CCI', 'CLAAS3']:
        raise Exception('Dataset {} not available!'.format(dataset))

    ofile_cma = 'CMA_{}_CALIOP_{}{}_DNT-{}_SATZ-{}.png'
    ofile_cph = 'CPH_{}_CALIOP_{}{}_DNT-{}_SATZ-{}.png'
    ofile_ctth = 'CTTH_{}_CALIOP_{}{}_DNT-{}_SATZ-{}.png'
    ofile_scat = 'SCATTER_{}_CALIOP_{}{}_DNT-{}_SATZ-{}.png'

    # iterate over satzen limitations
    for satz_lim in satzs:

        # if satz_lim list item is string convert it to float
        if satz_lim is not None:
            if isinstance(satz_lim, str):
                try:
                    satz_lim = float(satz_lim)
                except ValueError:
                    msg = 'Cannot convert {} to float'
                    raise Exception(msg.format(satz_lim))

        for dnt in dnts:

            dnt = dnt.upper()
            if dnt not in ['ALL', 'DAY', 'NIGHT', 'TWILIGHT']:
                raise Exception('DNT {} not recognized'.format(dnt))

            print('---------------------------')
            print('\nDNT = {}\n'.format(dnt))
            print('---------------------------')

            # set output filenames for CPH and CMA plot
            ofile_cma_tmp = ofile_cma.format(dataset, year, month, dnt, satz_lim)
            ofile_cph_tmp = ofile_cph.format(dataset, year, month, dnt, satz_lim)
            ofile_ctth_tmp = ofile_ctth.format(dataset, year, month, dnt, satz_lim)
            ofile_scat_tmp = ofile_scat.format(dataset, year, month, dnt, satz_lim)

            # get matchup data
            data, latlon = get_matchup_file_content(os.path.join(ipath, ifile),
                                                    chunksize, dnt, satz_lim,
                                                    dataset)

            # define plotting area
            module_path = os.path.dirname(__file__)
            adef = load_area(os.path.join(module_path, 'areas.yaml'),
                             plot_area)

            # for each input pixel get target pixel index
            resampler = BucketResampler(adef, latlon['lon'], latlon['lat'])
            idxs = resampler.idxs

            # get output grid size/lat/lon
            out_size = adef.size
            lon, lat = adef.get_lonlats()

            # do validation
            cma_scores = do_cma_cph_validation(data, adef, out_size,
                                               idxs, 'cma')
            cph_scores = do_cma_cph_validation(data, adef, out_size,
                                               idxs, 'cph')
            ctth_scores = do_ctth_validation(data, resampler, thrs=10)

            # get crs for plotting
            crs = adef.to_cartopy_crs()

            # get cos(lat) filed for weighted average on global regular grid
            cosfield = get_cosfield(lat)

            # do plotting
            make_plot(cma_scores, os.path.join(opath, ofile_cma_tmp), crs,
                      dnt, 'CMA', cosfield)
            make_plot(cph_scores, os.path.join(opath, ofile_cph_tmp), crs,
                      dnt, 'CPH', cosfield)
            make_plot_CTTH(ctth_scores, os.path.join(opath, ofile_ctth_tmp),
                           crs, dnt, 'CTTH', cosfield)
            make_scatter(data, os.path.join(opath, ofile_scat_tmp), dnt, dataset)
