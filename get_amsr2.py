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
        qual_cci = h5file[cci_key]['cpp_quality'][:].astype(int)

        if 'pixel_status' in h5file[amsr_key] and \
           'quality' in h5file[amsr_key] and \
           'surface_type' in h5file[amsr_key]:
            amsr_data_source = 'NSIDC'
            quality_amsr = h5file[amsr_key]['quality'][:]
            pixel_status_amsr = h5file[amsr_key]['pixel_status'][:]
            surface_type_amsr = h5file[amsr_key]['surface_type'][:]
        else:
            amsr_data_source = 'JAXA'


    N_NEIGHBOURS = lwp_cci.shape[-1]  # number of neighbours
    N_NEIGHBOURS_VALID = np.sum(img_linnum_nneigh > 0, axis=-1)
    BAD_QUALITY_PERC_MAX = 0.01  # maximum percentage of bad quality pixels
    VALID_NEIGHBOURS_PERC_MAX = 0.50  # minimum percentage of valid neighbours

    # quality checking using cci qcflag
    #bin(3) = 11000000, bit0 = Retrieval did not converge, bit1= Rettrieval cost > 100.
    bad_quality_img = np.bitwise_and(qual_cci, 3) > 0
    bad_quality_cnt_img = np.sum(bad_quality_img, axis=-1)
    use_quality_img = bad_quality_cnt_img/N_NEIGHBOURS_VALID < BAD_QUALITY_PERC_MAX

    # AMSR quality checking only available for NSIDC collocations
    if amsr_data_source == 'NSIDC':
        # only use amsr pixels where Quality Flag is 0
        use_quality_amsr = quality_amsr == 0
        # only use amsr pixels where pixel status is 0
        use_quality_amsr = np.logical_and(use_quality_amsr, pixel_status_amsr == 0)
        # only use amsr pixels where surface type is ocean (1)
        use_quality_amsr = np.logical_and(use_quality_amsr, surface_type_amsr == 1)

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
    mask_noneigh = img_linnum_nneigh < 0
    use_nvalid_neighbours = np.sum(mask_noneigh, axis=-1)/N_NEIGHBOURS < VALID_NEIGHBOURS_PERC_MAX

    # select valid values
    use_valid_values_amsr = np.logical_and(lwp_amsr >= 0, lwp_amsr < 170)
    use_valid_values_cci = np.logical_and(lwp_cci_mean >= 0, lwp_cci_mean < 3000)

    selection = np.logical_and(use_icf, use_landfrac)
    selection = np.logical_and(selection, use_sunz)
    selection = np.logical_and(selection, use_satz)
    selection = np.logical_and(selection, use_valid_values_amsr)
    selection = np.logical_and(selection, use_valid_values_cci)
    selection = np.logical_and(selection, use_nvalid_neighbours)
    selection = np.logical_and(selection, use_quality_img)
    selection = np.logical_and(selection, use_quality_amsr)

    return lwp_amsr[selection], lwp_cci_mean[selection], lat_amsr[selection], lon_amsr[selection]