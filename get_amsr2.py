import h5py
import numpy as np
from atrain_plot.get_imager import get_imager_cma
from atrain_plot.get_imager import get_imager_cph

def read_amsr_lwp(ifile, dataset):
    amsr_key = 'amsr'
    cci_key = dataset.lower()

    with h5py.File(ifile, 'r') as h5file:
        lwp_amsr = h5file[amsr_key]['lwp'][:]
        lwp_cci = h5file[cci_key]['cpp_lwp'][:]
        lat_amsr = h5file[amsr_key]['latitude'][:]
        lon_amsr = h5file[amsr_key]['longitude'][:]
        cph = get_imager_cph(h5file[cci_key])
        lsm = h5file[cci_key]['fractionofland'][:]
        sunz = h5file[cci_key]['sunz'][:]
        satz = h5file[cci_key]['satz'][:]
        img_linnum_nneigh = h5file[amsr_key]['imager_linnum_nneigh'][:]

    # mask pixels within footprint where no neighbour is found
    lwp_cci = np.where(img_linnum_nneigh <= 0, np.nan, lwp_cci)

    # set values < 0 to 0
    lwp_cci = np.where(lwp_cci < 0, 0, lwp_cci)

    # average imager LWP footprints
    lwp_cci_mean = np.nanmean(lwp_cci, axis=-1)

    # calculate ice cloud fraction (ICF) of footprint
    icf = np.nanmean(cph, axis=-1)
    # mask footprints where ICF >= 10%
    use_icf = icf < 0.1

    # calculate landfraction of footprints
    landfrac = np.nanmean(lsm, axis=-1)
    # mask pixels where land fraction > 0%
    use_landfrac = landfrac < 0.005

    # mask night and twilight
    use_sunz = sunz < 75

    # mask high zenith angles
    use_satz = satz < 70

    # minimum number of found neighbours limit
    if lwp_cci.shape[-1] == 5:
        N_NONEIGH_MAX = 3
    elif lwp_cci.shape[-1] == 8:
        N_NONEIGH_MAX = 5
    else:
        N_NONEIGH_MAX = lwp_cci.shape[-1]
    mask_noneigh = img_linnum_nneigh < 0
    use_nvalid_neighbours = np.sum(mask_noneigh, axis=-1) < N_NONEIGH_MAX

    # select valid values
    use_valid_values = np.logical_and(np.logical_and(lwp_amsr >= 0, lwp_amsr < 170),
                                      np.logical_and(lwp_cci_mean >= 0, lwp_cci_mean < 300))

    selection = np.logical_and(use_icf, use_landfrac)
    selection = np.logical_and(selection, use_sunz)
    selection = np.logical_and(selection, use_satz)
    selection = np.logical_and(selection, use_valid_values)
    selection = np.logical_and(selection, use_nvalid_neighbours)

    return lwp_amsr[selection], lwp_cci_mean[selection], lat_amsr[selection], lon_amsr[selection]