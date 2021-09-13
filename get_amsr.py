import h5py
import numpy as np
from atrain_plot.get_imager import get_imager_cma
from atrain_plot.get_imager import get_imager_cph

def read_amsr_lwp(ifile, dataset):
    amsr_key = 'amsr'
    cci_key = dataset.lower()

    with h5py.File(ifile, 'r') as h5file:

        amsr2 = {
            'lwp': h5file[amsr_key]['lwp'][:],
            'lat': h5file[amsr_key]['latitude'][:],
            'lon': h5file[amsr_key]['longitude'][:],
            'img_linnum_nneigh': h5file[amsr_key]['imager_linnum_nneigh'][:],
            'img_pixnum_nneigh': h5file[amsr_key]['imager_pixnum_nneigh'][:]
        }

        cci = {
            'lwp': h5file[cci_key]['cpp_lwp'][:],
            'cma': get_imager_cma(h5file[cci_key]),
            'cph': get_imager_cph(h5file[cci_key]),
            'lat': h5file[cci_key]['latitude'][:],
            'lon': h5file[cci_key]['longitude'][:],
            'satz': h5file[cci_key]['satz'][:],
            'sunz': h5file[cci_key]['sunz'][:],
            'lsm': h5file[cci_key]['fractionofland'][:]
        }

    lwp_amsr = amsr2['lwp']
    lwp_cci = cci['lwp']
    cma = cci['cma']
    cph = cci['cph']
    lsm = cci['lsm']
    sunz = cci['sunz']
    satz = cci['satz']
    img_linnum_nneigh = amsr2['img_linnum_nneigh']
    img_pixnum_nneigh = amsr2['img_pixnum_nneigh']

    Nstart = lwp_amsr.shape[0]

    # set values < 0 to 0
    lwp_cci = np.where(lwp_cci < 0, 0, lwp_cci)

    # mask outliers
    MAX_LWP = 300
    lwp_cci = np.where(lwp_cci > MAX_LWP, np.nan, lwp_cci)
    lwp_amsr = np.where(lwp_amsr > MAX_LWP, np.nan, lwp_amsr)

    # average LWP footprints
    lwp_cci_mean = np.nanmean(lwp_cci, axis=-1)
    # lwp_cci[lwp_cci == 0] = np.nan

    # calculate ice cloud fraction (ICF) of footprint
    icf = np.nanmean(cph, axis=-1)
    # mask footprints where ICF >= 20% (1 of 5 pixels)
    lwp_amsr = np.where(icf >= .2, np.nan, lwp_amsr)
    lwp_cci_mean = np.where(icf >= .2, np.nan, lwp_cci_mean)

    # calculate landfraction of footprints
    landfrac = np.nanmean(lsm, axis=-1)
    # mask pixels where land fraction > 10%
    lwp_amsr = np.where(landfrac > 0.0, np.nan, lwp_amsr)
    lwp_cci_mean = np.where(landfrac > 0.0, np.nan, lwp_cci_mean)

    # mask night and twilight
    lwp_amsr = np.where(sunz < 75, lwp_amsr, np.nan)
    lwp_cci_mean = np.where(sunz < 75, lwp_cci_mean, np.nan)

    # mask high zenith angles
    lwp_amsr = np.where(satz >= 70, np.nan, lwp_amsr)
    lwp_cci_mean = np.where(satz >= 70, np.nan, lwp_cci_mean)

    # mask collocations where we have no cci cloud
    lwp_amsr = np.where(np.sum(lwp_cci, axis=-1) <= 0, np.nan, lwp_amsr)
    lwp_cci_mean = np.where(np.sum(lwp_cci, axis=-1) <= 0, np.nan, lwp_cci_mean)

    # mask collocations where amsr LWP < 0.05
    lwp_amsr = np.where(lwp_amsr < 0.005, np.nan, lwp_amsr)
    lwp_cci_mean = np.where(lwp_amsr < 0.005, np.nan, lwp_cci_mean)

    # mask collocations where we have less than 3 valid cci measurements per footprint
    #N_INVALID_MAX = 2
    #lwp_amsr = np.where(np.sum(lwp_cci < 0, axis=-1) > N_INVALID_MAX,
    #                    np.nan, lwp_amsr)  # sum of invalid data !< 2
    #lwp_cci_mean = np.where(np.sum(lwp_cci < 0, axis=-1) > N_INVALID_MAX,
    #                        np.nan, lwp_cci_mean)  # sum of invalid data !< 2

    mask = np.logical_or(np.isnan(lwp_amsr), np.isnan(lwp_cci_mean))
    lwp_amsr = lwp_amsr[~mask]
    lwp_cci_mean = lwp_cci_mean[~mask]

    lat = amsr2['lat'][~mask]
    lon = amsr2['lon'][~mask]

    X = lwp_amsr  # [lwp_cci_mean > 0]
    Y = lwp_cci_mean  # [lwp_cci_mean > 0]

    return X, Y, lat, lon