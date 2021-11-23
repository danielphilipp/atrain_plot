import h5py
import numpy as np
from atrain_plot.get_imager import get_imager_cph


def _process_dardar_1_neighbour(cci, dardar, aux, variable):
    """ Process atrain_match DARDAR matchups with 1 neighbour. """
    
    cci['iwp'] = np.where(cci[variable] < 0, 0, cci[variable])
    dardar['iwp'] = np.sum(dardar['iwp'], axis=1)

    # quality checking using cci qcflag
    #bin(3) = 11000000, bit0 = Retrieval did not converge, bit1= Rettrieval cost > 100.
    bad_quality_img = np.bitwise_and(cci['quality'], 3) > 0
    bad_quality_img = np.where(dardar['img_linnum_nneigh'] < 0, np.nan, bad_quality_img)
    bad_quality_cnt_img = np.nansum(bad_quality_img, axis=-1)
    use_quality_img = bad_quality_cnt_img < 1

    use_valid_cci = np.logical_and(cci[variable] > 0, cci[variable] < 1000)
    use_valid_dardar = np.logical_and(dardar[variable] > 0, dardar[variable] < 3000)
    use_phase = cci['phase'] > 0.5
    use_satz = aux['satz'] < 70.
    selection = np.logical_and(use_phase, use_satz)
    selection = np.logical_and(selection, use_valid_cci)
    selection = np.logical_and(selection, use_quality_img)
    selection = np.logical_and(selection, use_valid_dardar)
    return cci[variable][selection], dardar[variable][selection], dardar['lat'][selection], dardar['lon'][selection]


def _process_dardar_n_neighbours(cci, dardar, aux, variable):
    """ Process atrain_match DARDAR matchups with multiple neighbours per footprint. """

    if variable == 'iwp':
        vmin = 0
        vmax = 3000
    elif variable == 'cer':
        vmin = 0
        vmax = 300
    else:
        raise Exception('Variable {} not known. Aborting!'.format(variable))

    # vertically integrate DARDAR IWP
    dardar[variable] = np.sum(dardar[variable], axis=1)

    # mask elements in footprint where no neighbour is found
    cci[variable][cci[variable] < 0] = np.nan
    # calculate footprint mean
    cci[variable+'_mean'] = np.nanmean(cci[variable], axis=1)

    # quality checking using cci qcflag
    #bin(3) = 11000000, bit0 = Retrieval did not converge, bit1= Rettrieval cost > 100.
    bad_quality_img = np.bitwise_and(cci['quality'], 3) > 0
    bad_quality_img = np.where(dardar['img_linnum_nneigh'] < 0, np.nan, bad_quality_img)
    bad_quality_cnt_img = np.nansum(bad_quality_img, axis=-1)
    use_quality_img = bad_quality_cnt_img < 1

    use_range_cci = np.logical_and(cci[variable+'_mean'] > vmin, cci[variable+'_mean'] < vmax)
    use_range_dardar = np.logical_and(dardar[variable] > vmin, dardar[variable] < vmax)
    use_valid_cci = ~np.isnan(cci[variable+'_mean'])
    use_valid_dardar = ~np.isnan(dardar[variable])

    if variable == 'iwp':
        icf = np.nanmean(cci['phase'])
        use_phase = icf > 0.8

    #for f in range(dardar[variable].shape[0]):
    #    print(cci[variable][f], '  ', cci[variable+'_mean'][f], '  ', dardar[variable][f])

    use_satz = aux['satz'] < 70.
    selection = np.logical_and(use_satz, use_range_cci)
    selection = np.logical_and(selection, use_range_dardar)
    if variable == 'iwp':
        selection = np.logical_and(selection, use_phase)
    selection = np.logical_and(selection, use_valid_cci)
    selection = np.logical_and(selection, use_quality_img)
    selection = np.logical_and(selection, use_valid_dardar)

    return cci[variable+'_mean'][selection], dardar[variable][selection], dardar['lat'][selection], dardar['lon'][selection]


def read_dardar_iwp(ifile, dataset):
    cci = {}
    dardar = {}
    aux = {}
    with h5py.File(ifile, 'r') as h5file:
        # imager variables
        cci['iwp'] = h5file[dataset.lower()]['cpp_iwp'][:]
        cci['phase'] = get_imager_cph(h5file[dataset.lower()])
        cci['quality'] = h5file[dataset.lower()]['cpp_quality'][:].astype(int)
        # truth variables
        dardar['iwp'] = h5file['dardar']['iwc'][:]
        dardar['lat'] = h5file['dardar']['latitude'][:]
        dardar['lon'] = h5file['dardar']['longitude'][:]
        dardar['img_linnum_nneigh'] = h5file['dardar']['imager_linnum_nneigh'][:]
        # aux variables
        aux['sunz'] = h5file[dataset.lower()]['sunz'][:]
        aux['satz'] = h5file[dataset.lower()]['satz'][:]

    # 1 neighbour per DARDAR footprint
    if len(cci['iwp'].shape) == 1:
        return _process_dardar_1_neighbour(cci, dardar, aux, variable='iwp')
    # multiple neighbours per DARDAR footprint
    else:
        return _process_dardar_n_neighbours(cci, dardar, aux, variable='iwp')


def read_dardar_cer(ifile, dataset):
    cci = {}
    dardar = {}
    aux = {}
    with h5py.File(ifile, 'r') as h5file:
        # imager variables
        cci['cer'] = h5file[dataset.lower()]['cpp_cer'][:]
        cci['phase'] = get_imager_cph(h5file[dataset.lower()])
        cci['quality'] = h5file[dataset.lower()]['cpp_quality'][:].astype(int)
        # truth variables
        dardar['cer'] = h5file['dardar']['effective_radius'][:] * 1E5
        dardar['lat'] = h5file['dardar']['latitude'][:]
        dardar['lon'] = h5file['dardar']['longitude'][:]
        dardar['img_linnum_nneigh'] = h5file['dardar']['imager_linnum_nneigh'][:]
        # aux variables
        aux['sunz'] = h5file[dataset.lower()]['sunz'][:]
        aux['satz'] = h5file[dataset.lower()]['satz'][:]

    # 1 neighbour per DARDAR footprint
    if len(cci['cer'].shape) == 1:
        return _process_dardar_1_neighbour(cci, dardar, aux, variable='cer')
    # multiple neighbours per DARDAR footprint
    else:
        return _process_dardar_n_neighbours(cci, dardar, aux, variable='cer')

