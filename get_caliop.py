""" Module containing function to read CALIPSO data from matchups. """

import numpy as np
from atrain_match.utils.get_flag_info import get_calipso_clouds_of_type_i_feature_classification_flags_one_layer as get_cal_flag
from atrain_match.utils import validate_cph_util as vcu


def get_caliop_cth(ds):
    """Get CALIOP CTH."""
    cth = np.array(ds['layer_top_altitude'])[:, 0]
    elev = np.array(ds['elevation'])
    # set FillValue to NaN, convert to m
    cth = np.where(cth == -9999, np.nan, cth * 1000.)
    # compute height above surface
    cth_surf = cth - elev
    return cth_surf


def get_caliop_ctt(ds):
    """Get CALIOP CTT."""
    ctt = np.array(ds['midlayer_temperature'])[:, 0]
    ctt = np.where(ctt == -9999, np.nan, ctt + 273.15)
    ctt = np.where(ctt < 0, np.nan, ctt)
    return ctt


def get_calipso_clouds_of_type_i(cflag, calipso_cloudtype=0):
    """Get CALIPSO clouds of type i from top layer."""
    # bits 10-12, start at 1 counting
    return get_cal_flag(cflag, calipso_cloudtype=calipso_cloudtype)


def get_calipso_low_clouds(cfalg):
    """Get CALIPSO low clouds."""
    # type 0, 1, 2, 3 are low cloudtypes
    calipso_low = np.logical_or(
        np.logical_or(
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=0),
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=1)),
        np.logical_or(
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=2),
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=3)))
    return calipso_low


def get_calipso_medium_clouds(cfalg):
    """Get CALIPSO medium clouds."""
    # type 4,5 are mid-level cloudtypes (Ac, As)
    calipso_high = np.logical_or(
        get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=4),
        get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=5))
    return calipso_high


def get_calipso_high_clouds(cfalg):
    """Get CALIPSO high clouds."""
    # type 6, 7 are high cloudtypes
    calipso_high = np.logical_or(
        get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=6),
        get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=7))
    return calipso_high


def get_calipso_op(cfalg):
    """Get CALIPSO opaque clouds."""
    # type 1, 2, 5, 7 are opaque cloudtypes
    calipso_low = np.logical_or(
        np.logical_or(
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=1),
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=2)),
        np.logical_or(
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=5),
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=7)))
    return calipso_low


def get_calipso_tp(cfalg):
    """Get CALIPSO semi-transparent clouds."""
    # type 0,3,4,6 transparent/broken
    calipso_low = np.logical_or(
        np.logical_or(
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=0),
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=3)),
        np.logical_or(
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=4),
            get_calipso_clouds_of_type_i(cfalg, calipso_cloudtype=6)))
    return calipso_low


def get_calipso_low_clouds_op(match_calipso):
    """Get CALIPSO low and opaque clouds."""
    # type 0, 1, 2, 3 are low cloudtypes
    calipso_low = np.logical_or(
        get_calipso_clouds_of_type_i(match_calipso, calipso_cloudtype=1),
        get_calipso_clouds_of_type_i(match_calipso, calipso_cloudtype=2))
    return calipso_low


def get_calipso_medium_and_high_clouds_tp(match_calipso):
    """Get CALIPSO medium transparent and high transparent clouds."""
    # type 0, 1, 2, 3 are low cloudtypes
    calipso_transp = np.logical_or(
        get_calipso_clouds_of_type_i(match_calipso, calipso_cloudtype=4),
        get_calipso_clouds_of_type_i(match_calipso, calipso_cloudtype=6))
    return calipso_transp


def get_caliop_cph(ds):
    """
    CALIPSO_PHASE_VALUES:   unknown=0,
                            ice=1,
                            water=2,
    """
    phase = vcu.get_calipso_phase_inner(ds['feature_classification_flags'],
                                        max_layers=10,
                                        same_phase_in_top_three_lay=True)
    mask = phase.mask
    phase = np.array(phase)
    phase = np.where(phase == 0, np.nan, phase)
    phase = np.where(phase == 2, 0, phase)
    phase = np.where(np.logical_or(phase == 1, phase == 3), 1, phase)

    phase = np.where(mask, np.nan, phase)
    return phase


def get_caliop_cma(ds):
    cfrac_limit = 0.5
    caliop_cma = np.array(ds['cloud_fraction']) > cfrac_limit
    return caliop_cma.astype(bool)