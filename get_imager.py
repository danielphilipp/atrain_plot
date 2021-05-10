""" Module containing function to read imager data from matchups. """

import numpy as np


def get_imager_cth(ds):
    """ Get imager CTH. """
    alti = np.array(ds['ctth_height'])
    # set FillValue to NaN
    alti = np.where(alti < 0, np.nan, alti)
    # alti = np.where(alti>45000, np.nan, alti)
    return alti


def get_imager_ctt(ds):
    """ Get imager CTT. """
    tempe = np.array(ds['ctth_temperature'])
    tempe = np.where(tempe < 0, np.nan, tempe)
    return tempe

def get_imager_ctp(ds):
    """ Get imager CTP. """
    pres = np.array(ds['ctth_pressure'])
    pres = np.where(pres < 0, np.nan, pres)
    return pres

def get_imager_cph(ds):
    """ Get imager CPH. """
    phase = np.array(ds['cpp_phase'])
    phase = np.where(phase < 0, np.nan, phase)
    phase = np.where(phase > 10, np.nan, phase)
    phase = np.where(phase == 0, np.nan, phase)
    phase = np.where(phase == 1, 0, phase)
    phase = np.where(phase == 2, 1, phase)
    return phase


def get_imager_cma(ds):
    """ Get imager CMA. """
    data = np.array(ds['cloudmask'])
    binary = np.where(data == 0, 0, 1)
    binary = np.where(data < 0, np.nan, binary)
    return binary.astype(bool)