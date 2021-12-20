import h5py
import numpy as np
from atrain_plot.get_imager import get_imager_cph


def _process_dardar_n_neighbours(imager, dardar, aux, variable, dataset, verbose=False):
    """ Process atrain_match DARDAR matchups with multiple neighbours per footprint. """

    # set valid range
    if variable == 'iwp':
        vmin = 0
        vmax = 2000
    elif variable == 'cer':
        vmin = 0
        vmax = 800
    else:
        raise Exception('Variable {} not known. Aborting!'.format(variable))

    if variable == 'iwp':
        # drop profiles where certain DARDAR flags occur
        is_invalid_profile = np.isin(dardar['mask'], [-2, 3, 8, 11, 12, 13, 14, 15])
        use_dardar_profile = ~np.any(is_invalid_profile, axis=-1)
        if verbose:
            print('Profiles dropped due to DARDAR mask: ',
                  100 * np.sum(~use_dardar_profile) / dardar['mask'].shape[0], '\n')
            print('Profile drop for each DARDAR mask: ')
            for r in range(-2, 16):
                print(r, '  ->   ', 100 * np.sum(np.any(dardar['mask'] == r, axis=-1))/dardar['mask'].shape[0])

    if variable == 'iwp':
        # vertically integrate DARDAR IWP
        dardar[variable] = np.nansum(dardar[variable], axis=1) * 60

    if variable == 'cer':
        # take column average cer
        dardar[variable] = np.nanmean(dardar[variable], axis=1)

    # mask imager elements in footprint where no neighbour is found
    imager[variable][imager[variable] < 0] = np.nan
    imager[variable][dardar['img_linnum_nneigh'] < 0] = np.nan
    # calculate imager footprint mean
    imager[variable+'_mean'] = np.nanmean(imager[variable], axis=1)

    if verbose:
        print('Values dropped if cutoff is applied to imager data')
        print('--------------------------------------')
        for k in range(500, 10000, 500):
            N = np.sum(imager[variable+'_mean'] > k)
            print('{} >> {:d} ({:.2f}%)'.format(k, N, 100 * N/imager[variable+'_mean'].shape[0]))
        print('--------------------------------------')

    if dataset.lower() == 'cci':
        # quality checking using cci qcflag
        #bin(3) = 11000000, bit0 = Retrieval did not converge, bit1= Rettrieval cost > 100.
        bad_quality_imager = np.bitwise_and(imager['quality'], 3) > 0
        bad_quality_imager = np.where(dardar['img_linnum_nneigh'] < 0, np.nan, bad_quality_imager)
        bad_quality_cnt_imager = np.nansum(bad_quality_imager, axis=-1)
        use_quality_imager = bad_quality_cnt_imager < 1

    # mask elements out of valid range
    use_range_imager = np.logical_and(imager[variable+'_mean'] >= vmin, imager[variable+'_mean'] <= vmax)
    use_range_dardar = np.logical_and(dardar[variable] >= vmin, dardar[variable] <= vmax)
    # mask elements where value is NaN
    use_valid_imager = ~np.isnan(imager[variable+'_mean'])
    use_valid_dardar = ~np.isnan(dardar[variable])

    # mask night and twilight
    use_sunz = aux['sunz'] < 75.
    # mask high satzen
    use_satz = aux['satz'] < 70.

    if variable == 'iwp':
        # mask imager elements where footprint ice cloud fraction is not 100%
        icf = np.nanmean(imager['phase'], axis=1)
        use_phase = icf > 0.9
        use_phase = np.logical_and(use_phase, ~np.isnan(use_phase))

    # combine all binary filters
    selection = np.logical_and(use_satz, use_range_imager)
    selection = np.logical_and(selection, use_range_dardar)
    if variable == 'iwp':
        selection = np.logical_and(selection, use_phase)
        selection = np.logical_and(selection, use_dardar_profile)
    selection = np.logical_and(selection, use_valid_imager)
    if dataset.lower() == 'cci':
        selection = np.logical_and(selection, use_quality_imager)
    selection = np.logical_and(selection, use_valid_dardar)
    selection = np.logical_and(selection, use_sunz)

    imager_v = imager[variable+'_mean'][selection]
    dardar_v = dardar[variable][selection]
    lat = dardar['lat'][selection]
    lon = dardar['lon'][selection]

    if verbose:
        s = 'Dropped with {}: {:.2f} {}'
        ntot = selection.shape[0]
        print(s.format('use_satz', 100 * np.sum(~use_satz) / ntot, ' %'))
        print(s.format('use_sunz', 100 * np.sum(~use_sunz) / ntot, ' %'))
        print(s.format('use_range_cci', 100 * np.sum(~use_range_imager) / ntot, ' %'))
        print(s.format('use_range_dardar', 100 * np.sum(~use_range_dardar) / ntot, ' %'))
        if variable == 'iwp':
            print(s.format('use_phase', 100 * np.sum(~use_phase) / ntot, ' %'))
            print(s.format('use_phase (not NaN)', 100 * np.sum(~use_phase & ~use_valid_imager) / ntot, ' %'))
            print(s.format('use_dardar_profile', 100 * np.sum(~use_dardar_profile) / ntot, ' %'))
        print(s.format('use_valid_cci', 100 * np.sum(~use_valid_imager) / ntot, ' %'))
        print(s.format('use_valid_dardar', 100 * np.sum(~use_valid_dardar) / ntot, ' %'))
        print(s.format('use_quality_img', 100 * np.sum(~use_quality_imager) / ntot, ' %'))

        print('Total N collocations before filtering: ', ntot)
        print('N collocations dropped: ', np.sum(~selection), ' (', 100 * (np.sum(~selection)/ntot), '%)')

    return dardar_v, imager_v, lat, lon


def read_dardar_iwp(ifile, dataset):
    imager = {}
    dardar = {}
    aux = {}
    with h5py.File(ifile, 'r') as h5file:
        # imager variables
        imager['iwp'] = h5file[dataset.lower()]['cpp_iwp'][:]
        imager['phase'] = get_imager_cph(h5file[dataset.lower()])
        if dataset.lower() == 'cci':
            imager['quality'] = h5file[dataset.lower()]['cpp_quality'][:].astype(int)
        # truth variables
        dardar['iwp'] = h5file['dardar']['iwc'][:] / 1E2
        dardar['lat'] = h5file['dardar']['latitude'][:]
        dardar['lon'] = h5file['dardar']['longitude'][:]
        dardar['mask'] = h5file['dardar']['DARMASK_Simplified_Categorization'][:]
        dardar['img_linnum_nneigh'] = h5file['dardar']['imager_linnum_nneigh'][:]
        # aux variables
        aux['sunz'] = h5file[dataset.lower()]['sunz'][:]
        aux['satz'] = h5file[dataset.lower()]['satz'][:]

    return _process_dardar_n_neighbours(imager, dardar, aux, 'iwp', dataset)


def read_dardar_cer(ifile, dataset):
    imager = {}
    dardar = {}
    aux = {}
    with h5py.File(ifile, 'r') as h5file:
        # imager variables
        imager['cer'] = h5file[dataset.lower()]['cpp_cer'][:]
        imager['phase'] = get_imager_cph(h5file[dataset.lower()])
        if dataset.lower() == 'cci':
            imager['quality'] = h5file[dataset.lower()]['cpp_quality'][:].astype(int)
        # truth variables
        dardar['cer'] = h5file['dardar']['effective_radius'][:] * 1E6
        dardar['lat'] = h5file['dardar']['latitude'][:]
        dardar['lon'] = h5file['dardar']['longitude'][:]
        dardar['img_linnum_nneigh'] = h5file['dardar']['imager_linnum_nneigh'][:]
        # aux variables
        aux['sunz'] = h5file[dataset.lower()]['sunz'][:]
        aux['satz'] = h5file[dataset.lower()]['satz'][:]

    return _process_dardar_n_neighbours(imager, dardar, aux, 'cer', dataset)

