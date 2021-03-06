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
from collections import OrderedDict
from atrain_plot.scores import ScoreUtils
import atrain_plot.get_imager as gi
import atrain_plot.get_caliop as gc
import atrain_plot.get_amsr2 as ga
import atrain_plot.get_dardar as gd
import atrain_plot.scatter_plots as scat
import atrain_plot.map_plots as mp
from scipy.stats import pearsonr


import warnings     # suppress warning of masked array/nanmean
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")


def _apply_dnt_mask(cal_cma, sev_cma, cal_cph, sev_cph, 
                    cal_cth, sev_cth, cal_ctt, sev_ctt,
                    cal_ctp, sev_ctp, mask):
    """ Apply DNT mask if [DAY, NIGHT, TWILIGHT]. 
        mask is None for ALL. """
    if mask is not None:
        cal_cma = da.where(mask, np.nan, cal_cma)
        sev_cma = da.where(mask, np.nan, sev_cma)
        cal_cph = da.where(mask, np.nan, cal_cph)
        sev_cph = da.where(mask, np.nan, sev_cph)
        cal_cth = da.where(mask, np.nan, cal_cth)
        sev_cth = da.where(mask, np.nan, sev_cth)
        cal_ctt = da.where(mask, np.nan, cal_ctt)
        sev_ctt = da.where(mask, np.nan, sev_ctt)
        cal_ctp = da.where(mask, np.nan, cal_ctp)
        sev_ctp = da.where(mask, np.nan, sev_ctp)
    
    results = (cal_cma, sev_cma, 
               cal_cph, sev_cph, 
               cal_cth, sev_cth,
               cal_ctt, sev_ctt,
               cal_ctp, sev_ctp)

    return results


def _get_calipso_matchup_file_content(ipath, chunksize, dnt='ALL',
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

    # get CTH, CTT, CTP
    sev_cth = da.from_array(gi.get_imager_cth(imager), chunks=chunksize)
    cal_cth = da.from_array(gc.get_caliop_cth(caliop), chunks=chunksize)
    sev_ctt = da.from_array(gi.get_imager_ctt(imager), chunks=chunksize)
    cal_ctt = da.from_array(gc.get_caliop_ctt(caliop), chunks=chunksize)
    cal_cflag = np.array(caliop['feature_classification_flags'][::, 0])
    if filter_stratospheric:
        tropo_height = da.from_array(gc.get_tropopause_height(caliop), chunks=chunksize)
    sev_ctp = da.from_array(gi.get_imager_ctp(imager), chunks=(chunksize))
    cal_ctp = da.from_array(gc.get_caliop_ctp(caliop), chunks=(chunksize))

    if dataset == 'CCI':
        sev_quality = da.from_array(imager['cpp_quality'][:].astype(int), chunks=chunksize)
        bad_quality_imager = np.bitwise_and(sev_quality, 3) > 0
        sev_cth[bad_quality_imager] = np.nan
        sev_ctt[bad_quality_imager] = np.nan

    # get CMA, CPH, VZA, SZA, LAT and LON
    sev_cph = da.from_array(gi.get_imager_cph(imager), chunks=chunksize)
    cal_cph = da.from_array(gc.get_caliop_cph(caliop), chunks=chunksize)
    cal_cma = da.from_array(gc.get_caliop_cma(caliop), chunks=chunksize)
    sev_cma = da.from_array(gi.get_imager_cma(imager), chunks=chunksize)
    satz = da.from_array(imager['satz'], chunks=chunksize)
    sunz = da.from_array(imager['sunz'], chunks=chunksize)
    lat = da.from_array(imager['latitude'], chunks=chunksize)
    lon = da.from_array(imager['longitude'], chunks=chunksize)
    # mask fill values for angles
    satz = np.where(satz==-9, np.nan, satz)
    sunz = np.where(sunz==-9, np.nan, sunz)
    
    # mask satellize zenith angle
    if satz_lim is not None:
        mask = satz > satz_lim
        cal_cma = da.where(mask, np.nan, cal_cma)
        sev_cma = da.where(mask, np.nan, sev_cma)
        cal_cph = da.where(mask, np.nan, cal_cph)
        sev_cph = da.where(mask, np.nan, sev_cph)
        cal_cth = da.where(mask, np.nan, cal_cth)
        sev_cth = da.where(mask, np.nan, sev_cth)
        cal_ctt = da.where(mask, np.nan, cal_ctt)
        sev_ctt = da.where(mask, np.nan, sev_ctt)
        cal_ctp = da.where(mask, np.nan, cal_ctp)
        sev_ctp = da.where(mask, np.nan, sev_ctp)
    # mask all pixels except daytime
    if dnt == 'DAY':
        mask = sunz >= 80
    # mask all pixels except nighttime
    elif dnt == 'NIGHT':
        mask = sunz <= 95
    # mask all pixels except twilight
    elif dnt == 'TWILIGHT':
        mask = ~da.logical_and(sunz > 80, sunz < 95)
    # no masking
    elif dnt == 'ALL':
        mask = None
    else:
        raise Exception('DNT option ', dnt, ' is invalid.')

    # filter stratospheric clouds in CALIPSO 
    if filter_stratospheric:
        strato_mask = cal_cth>tropo_height
        cal_cma = da.where(strato_mask, np.nan, cal_cma)
        cal_cph = da.where(strato_mask, np.nan, cal_cph)
        cal_cth = da.where(strato_mask, np.nan, cal_cth)
        cal_ctt = da.where(strato_mask, np.nan, cal_ctt)
        cal_ctp = da.where(strato_mask, np.nan, cal_ctp)
    
    # apply DNT masking
    masked = _apply_dnt_mask(cal_cma, sev_cma, cal_cph, 
                             sev_cph, cal_cth, sev_cth,
                             cal_ctt, sev_ctt,
                             cal_ctp, sev_ctp,
                             mask)
    
    data = {'caliop_cma': masked[0],
            'imager_cma': masked[1],
            'caliop_cph': masked[2],
            'imager_cph': masked[3],
            'satz': satz,
            'sunz': sunz,
            'caliop_cth': masked[4],
            'imager_cth': masked[5],
            'caliop_ctt': masked[6],
            'imager_ctt': masked[7],
            'caliop_ctp': masked[8],
            'imager_ctp': masked[9],
            'caliop_cflag': cal_cflag,
            }

    latlon = {'lat': lat,
              'lon': lon}

    return data, latlon


def _do_cma_cph_validation(data, adef, out_size, idxs, variable):
    """ Calculate scores for CMA or CPH depending on variable arg. """
    
    # !!! for CPH '0' is water, '1' is ice !!!
    
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
    if variable.lower() == 'cph' : scores['PODwater'] = scores.pop('PODclr')
    scores['PODcld'] = [scu.pod_1().reshape(adef.shape),
                        0.5, 1, 'rainbow']
    if variable.lower() == 'cph': scores['PODice'] = scores.pop('PODcld')
    scores['FARclr'] = [scu.far_0().reshape(adef.shape),
                        0, 1, 'rainbow']
    if variable.lower() == 'cph': scores['FARwater'] = scores.pop('FARclr')
    scores['FARcld'] = [scu.far_1().reshape(adef.shape),
                        0, 1, 'rainbow']
    if variable.lower() == 'cph': scores['FARice'] = scores.pop('FARcld')
    scores['POFDclr'] = [scu.pofd_0().reshape(adef.shape),
                         0, 1, 'rainbow']
    if variable.lower() == 'cph': scores['POFDwater'] = scores.pop('POFDclr')
    scores['POFDcld'] = [scu.pofd_1().reshape(adef.shape),
                         0, 1, 'rainbow']
    if variable.lower() == 'cph': scores['POFDice'] = scores.pop('POFDcld')
    scores['Heidke'] = [scu.heidke().reshape(adef.shape),
                        0, 1, 'rainbow']
    scores['Kuiper'] = [scu.kuiper().reshape(adef.shape),
                        0, 1, 'rainbow']
    scores['Bias'] = [scu.bias().reshape(adef.shape),
                      0, 1, 'bwr']
    scores['CALIOP mean'] = [scu.mean(a, c).reshape(adef.shape),
                             None, None, 'rainbow']
    scores['SEVIRI mean'] = [scu.mean(a, b).reshape(adef.shape),
                             None, None, 'rainbow']
    scores['Nobs'] = [scu.n.reshape(adef.shape),
                      None, None, 'rainbow']

    # calculate bias limits for plotting
    scores['Bias'][2] = np.nanmax(np.abs(scores['Bias'][0])) / 2
    scores['Bias'][1] = scores['Bias'][2] * (-1)
    
    # change mean values, bias
    if variable.lower() == 'cph':
        scores['CALIOP mean'][0] = scu.mean(d, b).reshape(adef.shape)
        scores['SEVIRI mean'][0] = scu.mean(d, c).reshape(adef.shape)
        scores['Bias'][0] = scores['Bias'][0] * (-1)
  
    return scores


def _do_ctth_validation_OLD(data, resampler, thrs=10):
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


def _do_ctth_validation(data, adef, out_size, idxs, resampler, thrs=10):
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
    detected_pressure = np.logical_and(detected_clouds,
                                          np.isfinite(data['imager_ctp']))                                      
    detected_height_mask = detected_height.astype(int)

    # calculate bias and mea for all ctth cases
    delta_h = data['imager_cth'] - data['caliop_cth']  # HEIGHT
    height_bias = np.where(detected_height, delta_h, np.nan)
    delta_t = data['imager_ctt'] - data['caliop_ctt']  # TEMPERATURE
    temperature_bias = np.where(detected_temperature, delta_t, np.nan)
    delta_p = data['imager_ctp'] - data['caliop_ctp']  # PRESSURE
    pressure_bias = np.where(detected_pressure, delta_p, np.nan)

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

    # resample and filter some data out
    n_matched_cases = resampler.get_sum(detected_height_mask)
    sev_cth_average = resampler.get_average(data['imager_cth'])
    cal_cth_average = resampler.get_average(data['caliop_cth'])
    bias_average = resampler.get_average(height_bias)
    bias_average = np.where(n_matched_cases < thrs, np.nan, bias_average)
    bias_temperature_average = resampler.get_average(temperature_bias)
    bias_temperature_average = np.where(n_matched_cases < thrs, np.nan,
                                        bias_temperature_average)
    bias_pressure_average = resampler.get_average(pressure_bias)
    bias_pressure_average = np.where(n_matched_cases < thrs, np.nan,
                                        bias_pressure_average)

    n_matched_cases_low = resampler.get_sum(detected_low.astype(int))
    bias_low_average = resampler.get_average(bias_low)
    bias_low_average = np.where(n_matched_cases_low < thrs,
                                np.nan, bias_low_average)
    n_matched_cases_mid = resampler.get_sum(detected_mid.astype(int))
    bias_mid_average = resampler.get_average(bias_mid)
    bias_mid_average = np.where(n_matched_cases_mid < thrs, np.nan,
                                bias_mid_average)
    n_matched_cases_high = resampler.get_sum(detected_high.astype(int))
    bias_high_average = resampler.get_average(bias_high)
    bias_high_average = np.where(n_matched_cases_high < thrs, np.nan,
                                 bias_high_average)

    # calculate scores
    scores = dict()
    # [scores_on_target_grid, vmin, vmax, cmap]
    scores['Bias CTH'] = [bias_average, -4000, 4000, 'bwr']
    scores['Bias CTT'] = [bias_temperature_average, -30, 30, 'bwr']
    scores['Bias CTP'] = [bias_pressure_average, -200, 200, 'bwr']
    
    scores['Bias CTH low'] = [bias_low_average, -2000, 2000, 'bwr']
    scores['Bias CTH middle'] = [bias_mid_average, -2000, 2000, 'bwr']
    scores['Bias CTH high'] = [bias_high_average, -6000, 6000, 'bwr']
    
    addit_scores = _do_ctp_validation(data, adef, out_size, idxs)
    scores.update(addit_scores)
    
    scores['CALIOP CTH mean'] = [cal_cth_average, 1000, 14000, 'rainbow']
    scores['SEVIRI CTH mean'] = [sev_cth_average, 1000, 14000, 'rainbow']
    scores['Num_detected_height'] = [n_matched_cases, None, None, 'rainbow']
    
    return scores


def _do_ctp_validation(data, adef, out_size, idxs):
    """ Calculate CTP validation (included in CTTH plot). """

    # detected ctth mask
    detected_clouds = da.logical_and(data['caliop_cma'] == 1,
                                     data['imager_cma'] == 1)
    detected_height = da.logical_and(detected_clouds,
                                     np.isfinite(data['imager_cth']))
    # find pps low and caliop low
    low_clouds_c = gc.get_calipso_low_clouds(data['caliop_cflag'])
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

    scu = ScoreUtils(a, b, c, d)
    scores = dict()
    scores['CTP low clouds POD'] = [scu.pod_1().reshape(adef.shape), 0, 1, 'rainbow']
    scores['CTP low clouds FAR'] = [scu.far_1().reshape(adef.shape), 0, 1, 'rainbow']
    scores['CTP low clouds POFD'] = [scu.pofd_1().reshape(adef.shape), 0, 1, 'rainbow']
    # scores['Heidke low clouds'] = [scu.heidke().reshape(adef.shape),0, 1, 'rainbow']

    return scores


def _get_cosfield(lat):
    """ Calculate 2D cos(lat) field for weighted averaging on regular grid."""

    latcos = np.abs(np.cos(lat * np.pi / 180))
    cosfield = da.from_array(latcos, chunks=(1000, 1000))  # [mask]
    return cosfield


def _weighted_spatial_average(data, cosfield):
    """ Calculate weighted spatial average. """

    if isinstance(data, xr.DataArray):
        data = data.data
    if isinstance(data, np.ndarray):
        data = da.from_array(data, chunks=(1000, 1000))
    return da.nansum(data * cosfield) / da.nansum(cosfield)


def _gaussian(bins, mu, sigma):
   return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ) 


def _process_cer_or_lwp_or_iwp(truth_xwp, imager_xwp, truth_lat, truth_lon, adef, var, dataset, chunksize):
    truth_xwp = da.from_array(truth_xwp, chunks=chunksize)
    imager_xwp = da.from_array(imager_xwp, chunks=chunksize)
    truth_lat = da.from_array(truth_lat, chunks=chunksize)
    truth_lon = da.from_array(truth_lon, chunks=chunksize)

    # for each input pixel get target pixel index
    resampler = BucketResampler(adef, truth_lon, truth_lat)

    truth_mean = resampler.get_average(truth_xwp)
    imager_mean = resampler.get_average(imager_xwp)
    bias = resampler.get_average(imager_xwp - truth_xwp)
    bcrmse = np.sqrt(resampler.get_average((imager_xwp - truth_xwp - np.mean((imager_xwp - truth_xwp))) ** 2))
    rmse = np.sqrt(resampler.get_average((imager_xwp - truth_xwp) ** 2))

    variable = var.split(' ')[1]

    results = {'{} mean'.format(var): [truth_mean.reshape(adef.shape), 0, 100, 'rainbow'],
               '{} {} mean'.format(dataset, variable): [imager_mean.reshape(adef.shape), 0, 100, 'rainbow'],
               '{} Bias'.format(variable): [bias.reshape(adef.shape), -30, 30, 'bwr'],
               '{} BC-RMSE'.format(variable): [bcrmse.reshape(adef.shape), None, None, 'rainbow'],
               '{} RMSE'.format(variable): [rmse.reshape(adef.shape), 0, 100, 'rainbow'],
               '{} Nobs'.format(variable): [resampler.get_count(), None, None, 'rainbow']}

    return results


def _print_opts(ifilepath_calipso, ifilepath_amsr, ifilepath_dardar, opath, dnts, satzs,
                dataset, filter_stratospheric):
    print('\n################ OPTIONS ################\n')
    print('Output path: ', opath)
    print('DNTs: ', dnts)
    print('Satellize Zenith limits: ', satzs)
    print('Imager dataset: ', dataset)

    if ifilepath_calipso is not None:
        print('Processing CALIPSO CFC/CPH/CTH: YES')
    else:
        print('Processing CALIPSO CFC/CPH/CTH: NO')

    if ifilepath_amsr is not None:
        print('Processing AMSR2 LWP: YES')
    else:
        print('Processing AMSR2 LWP: NO')

    if ifilepath_dardar is not None:
        print('Processing DARDAR IWP: YES')
    else:
        print('Processing DARDAR IWP: NO')

    print('Filter stratospheric clouds: ', filter_stratospheric)
    print('\n#########################################\n')


def run(opath, year, month, dataset, ifilepath_calipso=None, ifilepath_amsr=None,
        ifilepath_dardar=None, satzs=[None], dnts=['ALL', 'DAY', 'NIGHT', 'TWILIGHT'],
        chunksize=100000, plot_area='pc_world', FILTER_STRATOSPHERIC_CLOUDS=False):
    """
    Main function to be called in external script to run atrain_plot.

    opath (str):             Path where figures should be saved.
    year (str):              String of year.
    month (str):             String of month.
    dataset (str):           Dataset validated: Available: CCI, CLAAS.
    ifilepath_calipso (str): Path to CALIPSO HDF5 atrain_match matchup file.
    ifilepath_amsr (str):    Path to CALIPSO HDF5 atrain_match matchup file.
    ifilepath_dardar (str):  Path to CALIPSO HDF5 atrain_match matchup file.
    satzs (list):            List of satellite zenith limitations to be processed. Use
                             [None] if no limitation required.
    dnts (list):             List of illumination scenarios to be processed.
                             Available: ALL, DAY, NIGHT, TWILIGHT
    chunksize (int):         Size of data chunks to reduce memory usage.
    plot_area (str):         Name of area definition in areas.yaml file to be used.
    FILTER_STRATOSPHERIC_CLOUDS(bool): filter calipso data with CTH>tropopause height
    """

    global filter_stratospheric
    filter_stratospheric = FILTER_STRATOSPHERIC_CLOUDS

    if ifilepath_calipso is None and ifilepath_amsr is None and ifilepath_dardar is None:
        raise Exception('All input files (CALIOP/AMSR/DARDAR) are None. Not running validation!')

    _print_opts(ifilepath_calipso, ifilepath_amsr, ifilepath_dardar, opath, dnts, satzs,
                dataset, filter_stratospheric)
    
    # if dnts is single string convert to list
    if isinstance(dnts, str):
        dnts = [dnts]

    # if satzs is single string/int/float convert to list
    if isinstance(satzs, str) or isinstance(satzs, int) or isinstance(satzs, float):
        satzs = [satzs]

    if dataset not in ['CCI', 'CLAAS']:
        raise Exception('Dataset {} not available!'.format(dataset))

    # define plotting area
    module_path = os.path.dirname(__file__)
    adef = load_area(os.path.join(module_path, 'areas.yaml'),
                     plot_area)

    # get output grid size/lat/lon
    out_size = adef.size
    lon, lat = adef.get_lonlats()

    # get crs for plotting
    crs = adef.to_cartopy_crs()

    ofile_cma = 'CMA_{}_CALIOP_{}{}_DNT-{}_SATZ-{}.png'
    ofile_cph = 'CPH_{}_CALIOP_{}{}_DNT-{}_SATZ-{}.png'
    ofile_ctth = 'CTTH_{}_CALIOP_{}{}_DNT-{}_SATZ-{}.png'
    ofile_scat = 'CTX_SCATTER_{}_CALIOP_{}{}_DNT-{}_SATZ-{}.png'

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

        print('------------- SATZ = {} -------------'.format(satz_lim))

        for dnt in dnts:

            dnt = dnt.upper()
            if dnt not in ['ALL', 'DAY', 'NIGHT', 'TWILIGHT']:
                raise Exception('DNT {} not recognized'.format(dnt))

            print('---------------------------')
            print('DNT = {}'.format(dnt))
            print('---------------------------')

            if ifilepath_calipso is not None:
                # set output filenames for CPH and CMA plot
                ofile_cma_tmp = ofile_cma.format(dataset, year, month, dnt, satz_lim)
                ofile_cph_tmp = ofile_cph.format(dataset, year, month, dnt, satz_lim)
                ofile_ctth_tmp = ofile_ctth.format(dataset, year, month, dnt, satz_lim)
                ofile_scat_tmp = ofile_scat.format(dataset, year, month, dnt, satz_lim)

                # get matchup data
                data, latlon = _get_calipso_matchup_file_content(ifilepath_calipso,
                                                         chunksize, dnt, satz_lim,
                                                         dataset)

                # for each input pixel get target pixel index
                resampler = BucketResampler(adef, latlon['lon'], latlon['lat'])
                idxs = resampler.idxs

                # do validation
                cma_scores = _do_cma_cph_validation(data, adef, out_size,
                                                    idxs, 'cma')
                cph_scores = _do_cma_cph_validation(data, adef, out_size,
                                                    idxs, 'cph')
                ctth_scores = _do_ctth_validation(data, adef, out_size,
                                                    idxs, resampler, thrs=10)


                # get cos(lat) filed for weighted average on global regular grid
                cosfield = _get_cosfield(lat)

                # do plotting
                mp.make_plot_cma_cph(cma_scores, os.path.join(opath, ofile_cma_tmp), crs,
                           dnt, 'CMA', cosfield)
                mp.make_plot_cma_cph(cph_scores, os.path.join(opath, ofile_cph_tmp), crs,
                           dnt, 'CPH', cosfield)
                mp.make_plot_CTTH(ctth_scores, os.path.join(opath, ofile_ctth_tmp),
                                crs, dnt, 'CTTH', cosfield)
                scat.make_scatter_ctx(data, os.path.join(opath, ofile_scat_tmp), dnt, dataset)

    if ifilepath_amsr is not None:
        # LWP plots
        amsr_lwp, imager_lwp, amsr_lat, amsr_lon = ga.read_amsr_lwp(ifilepath_amsr, dataset)

        ofile_lwp = 'LWP_{}_AMSR2_{}{}.png'.format(dataset, year, month)
        ofile_lwp_scatter = 'LWP_SCATTER_{}_AMSR2_{}{}.png'.format(dataset, year, month)

        results = _process_cer_or_lwp_or_iwp(amsr_lwp, imager_lwp, amsr_lat, amsr_lon,
                                      adef, 'AMSR2 LWP', dataset, chunksize)

        mp.make_plot_lwp(results, crs, os.path.join(opath, ofile_lwp))
        scat.make_scatter_lwp(amsr_lwp, imager_lwp, os.path.join(opath, ofile_lwp_scatter))

    if ifilepath_dardar is not None:
        # IWP plots
        dardar_iwp, imager_iwp, dardar_lat, dardar_lon = gd.read_dardar_iwp(ifilepath_dardar, dataset)

        ofile_iwp = 'IWP_{}_DARDAR_{}{}.png'.format(dataset, year, month)
        ofile_iwp_scatter = 'IWP_SCATTER_{}_DARDAR_{}{}.png'.format(dataset, year, month)

        results = _process_cer_or_lwp_or_iwp(dardar_iwp, imager_iwp, dardar_lat, dardar_lon,
                                      adef, 'DARDAR IWP', dataset, chunksize)

        mp.make_plot_iwp(results, crs, os.path.join(opath, ofile_iwp))
        scat.make_scatter_iwp(dardar_iwp, imager_iwp, os.path.join(opath, ofile_iwp_scatter))

        # CER plots
        dardar_cer, imager_cer, dardar_lat, dardar_lon = gd.read_dardar_cer(ifilepath_dardar, dataset)

        ofile_cer = 'CER_{}_DARDAR_{}{}.png'.format(dataset, year, month)
        ofile_cer_scatter = 'CER_SCATTER_{}_DARDAR_{}{}.png'.format(dataset, year, month)

        results = _process_cer_or_lwp_or_iwp(dardar_cer, imager_cer, dardar_lat, dardar_lon,
                                      adef, 'DARDAR CER', dataset, chunksize)

        mp.make_plot_cer(results, crs, os.path.join(opath, ofile_cer))
        scat.make_scatter_cer(dardar_cer, imager_cer, os.path.join(opath, ofile_cer_scatter))
