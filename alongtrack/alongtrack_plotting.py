import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.ma as ma
from datetime import datetime
from pandas import DataFrame, Index
import glob
import tarfile
import bz2
import satpy
from pyorbital import astronomy
from rgb import calc_rgb_with016
import cartopy.crs as ccrs
import pyresample


matplotlib.use('Agg')


class CalipsoCollocations():
    """ Class containing CALIPSO collocated data for plotting. """

    def __init__(self):
        self.sfcIce = list()
        self.sfcTop = list()
        self.sfcTyp = list()

        self.cod = list()

        self.phase0 = list()
        self.phase1 = list()
        self.phase2 = list()

        self.type0 = list()
        self.type1 = list()
        self.type2 = list()

        self.ctp0Top = list()
        self.ctp1Top = list()
        self.ctp2Top = list()

        self.ctp0Bot = list()
        self.ctp1Bot = list()
        self.ctp2Bot = list()

        self.ctt0Top = list()
        self.ctt1Top = list()
        self.ctt2Top = list()

        self.cth0Top = list()
        self.cth1Top = list()
        self.cth2Top = list()

        self.cth0Bot = list()
        self.cth1Bot = list()
        self.cth2Bot = list()

        self.lat0 = list()
        self.lat1 = list()
        self.lat2 = list()

        self.lon0 = list()
        self.lon1 = list()
        self.lon2 = list()

        self.time = list()

    def lists_to_arrays(self):
        self.sfcIce = np.array(self.sfcIce)
        self.sfcTop = np.array(self.sfcTop)
        self.sfcTyp = np.array(self.sfcTyp)

        self.cod = np.array(self.cod)

        self.phase0 = np.array(self.phase0)
        self.phase1 = np.array(self.phase1)
        self.phase2 = np.array(self.phase2)

        self.type0 = np.array(self.type0)
        self.type1 = np.array(self.type1)
        self.type2 = np.array(self.type2)

        self.ctp0Top = np.array(self.ctp0Top)
        self.ctp1Top = np.array(self.ctp1Top)
        self.ctp2Top = np.array(self.ctp2Top)

        self.ctp0Bot = np.array(self.ctp0Bot)
        self.ctp1Bot = np.array(self.ctp1Bot)
        self.ctp2Bot = np.array(self.ctp2Bot)

        self.ctt0Top = np.array(self.ctt0Top)
        self.ctt1Top = np.array(self.ctt1Top)
        self.ctt2Top = np.array(self.ctt2Top)

        self.cth0Top = np.array(self.cth0Top)
        self.cth1Top = np.array(self.cth1Top)
        self.cth2Top = np.array(self.cth2Top)

        self.cth0Bot = np.array(self.cth0Bot)
        self.cth1Bot = np.array(self.cth1Bot)
        self.cth2Bot = np.array(self.cth2Bot)

        self.lat0 = np.array(self.lat0)
        self.lat1 = np.array(self.lat1)
        self.lat2 = np.array(self.lat2)

        self.lon0 = np.array(self.lon0)
        self.lon1 = np.array(self.lon1)
        self.lon2 = np.array(self.lon2)

        self.time = np.array(self.time)


class ImagerCollocations:
    """ Class containing imager collocated data for plotting. """

    def __init__(self):
        self.ctp = list()
        self.cth = list()
        self.ctt = list()
        self.cph = list()

    def lists_to_arrays(self):
        self.ctp = np.array(self.ctp)
        self.ctp = ma.masked_greater(self.ctp, 10000.)

        self.cth = np.array(self.cth)
        self.cth = ma.masked_greater(self.cth, 100000.)

        self.ctt = np.array(self.ctt)
        self.ctt = ma.masked_greater(self.ctt, 10000.)

        self.cph = np.array(self.cph)


def _intToBin(x):
    """ Convert integer to binary. """

    return str(bin(x)[2:])


def _correct_calipso_phase(phase_in):
    """ Small changes to calipso phase. """

    phase_out = np.array(phase_in)
    phase_out = phase_out.astype(float)
    phase_out[phase_out == 1] = 999.
    phase_out[phase_out == 2] = 1.
    phase_out[phase_out == 999.] = 2.
    phase_out[phase_out == 3] = 2.
    phase_out[phase_out == 0.] = 999.
    return phase_out


def _correct_cci_phase(phase):
    """ Small changes to imager phase. """

    phase[phase == 0.] = np.nan
    phase[phase > 2.] = np.nan
    return phase


def _get_collocated_file_data(filepath, idx0, idx1):
    """ Read atrain_match collocation file. """

    file = h5py.File(filepath, 'r')

    cal = file['calipso']
    img = file['cci']

    data_cal = {'lat': cal['latitude'][idx0:idx1],
                'lon': cal['longitude'][idx0:idx1],
                'colCOD': cal['column_optical_depth_cloud_532'][idx0:idx1],
                'featCOD': cal['feature_optical_depth_532'][idx0:idx1, :],
                'topCTP': cal['layer_top_pressure'][idx0:idx1, :],
                'botCTP': cal['layer_base_pressure'][idx0:idx1, :],
                'topCTT': cal['layer_top_temperature'][idx0:idx1, :],
                'botCTT': cal['midlayer_temperature'][idx0:idx1, :],
                'topCTH': cal['layer_top_altitude'][idx0:idx1, :],
                'botCTH': cal['layer_base_altitude'][idx0:idx1, :],
                'cFlag': cal['feature_classification_flags'][idx0:idx1, :],
                'sfcIce': cal['nsidc_surface_type'][idx0:idx1],
                'sfcType': cal['igbp_surface_type'][idx0:idx1],
                'sfcElev': cal['elevation'][idx0:idx1],
                'time': cal['sec_1970'][idx0:idx1]
                }

    data_img = {'cma': img['cloudmask'][idx0:idx1],
                'cph': img['cpp_phase'][idx0:idx1].astype(np.float32),
                'cth': img['ctth_height'][idx0:idx1],
                'cth_c': img['ctth_height_corr'][idx0:idx1],
                'ctt': img['ctth_temperature'][idx0:idx1],
                'ctp': img['ctth_pressure'][idx0:idx1],
                'cth_asl': img['imager_ctth_m_above_seasurface'][idx0:idx1]
                }

    return data_cal, data_img


def _get_cflag(flag, idx1, idx2):
    """ Get cloud flag value from binary flag. """

    flag_int = int(flag)
    flag_len = flag_int.bit_length()
    flag_bin = _intToBin(flag)
    return int(flag_bin[flag_len - idx1:flag_len - idx2], 2)


def _prepare_data(calipso, imager):
    """ Prepare collocated data for crosssection alongtrack plot. """

    n_collocations = calipso['lat'].shape[0]
    nlayers = calipso['cFlag'].shape[1]

    colCal = CalipsoCollocations()
    colImg = ImagerCollocations()

    for i in range(n_collocations):
        firstLayerFound = False
        secondLayerFound = False
        thirdLayerFound = False

        colCal.sfcIce.append(calipso['sfcIce'][i])
        colCal.sfcTop.append(calipso['sfcElev'][i])
        colCal.sfcTyp.append(calipso['sfcType'][i])
        colCal.time.append(calipso['time'][i])

        colCodCalSum = 0.

        colImg.ctt.append(imager['ctt'][i])
        colImg.ctp.append(imager['ctp'][i])
        colImg.cth.append(imager['cth'][i])
        colImg.cph.append(imager['cph'][i])

        for lay in range(nlayers):
            cflag_tmp = calipso['cFlag'][i, lay]
            colCmaskCal = _get_cflag(cflag_tmp, 3, 0)

            if colCmaskCal == 2:
                colCodCalSum += calipso['featCOD'][i, lay]

                # first layer
                if colCodCalSum > 0. and not firstLayerFound:
                    firstLayerFound = True
                    # append COD
                    colCal.cod.append(calipso['colCOD'][i])
                    # append phase and type
                    colCal.phase0.append(_get_cflag(cflag_tmp, 7, 5))
                    colCal.type0.append(_get_cflag(cflag_tmp, 12, 9))
                    # append CTP, CTH and CTT
                    colCal.ctp0Top.append(calipso['topCTP'][i, lay])
                    colCal.ctp0Bot.append(calipso['botCTP'][i, lay])
                    colCal.cth0Top.append(calipso['topCTH'][i, lay])
                    colCal.cth0Bot.append(calipso['botCTH'][i, lay])
                    colCal.ctt0Top.append(calipso['topCTT'][i, lay] + 273.15)

                # second layer
                if colCodCalSum > 0.15 and not secondLayerFound:
                    secondLayerFound = True
                    # append phase and type
                    colCal.phase1.append(_get_cflag(cflag_tmp, 7, 5))
                    colCal.type1.append(_get_cflag(cflag_tmp, 12, 9))
                    # append CTP, CTH and CTT
                    colCal.ctp1Top.append(calipso['topCTP'][i, lay])
                    colCal.ctp1Bot.append(calipso['botCTP'][i, lay])
                    colCal.cth1Top.append(calipso['topCTH'][i, lay])
                    colCal.cth1Bot.append(calipso['botCTH'][i, lay])
                    colCal.ctt1Top.append(calipso['topCTT'][i, lay] + 273.15)
                    colCal.lat1.append(calipso['lat'][i])
                    colCal.lon1.append(calipso['lon'][i])

                # third layer
                if colCodCalSum > 1. and not thirdLayerFound:
                    thirdLayerFound = True
                    # append phase and type
                    colCal.phase2.append(_get_cflag(cflag_tmp, 7, 5))
                    colCal.type2.append(_get_cflag(cflag_tmp, 12, 9))
                    # append CTP, CTH and CTT
                    colCal.ctp2Top.append(calipso['topCTP'][i, lay])
                    colCal.ctp2Bot.append(calipso['botCTP'][i, lay])
                    colCal.cth2Top.append(calipso['topCTH'][i, lay])
                    colCal.cth2Bot.append(calipso['botCTH'][i, lay])
                    colCal.ctt2Top.append(calipso['topCTT'][i, lay] + 273.15)
                    colCal.lat2.append(calipso['lat'][i])
                    colCal.lon2.append(calipso['lon'][i])

            if lay == nlayers-1:
                colCal.lat0.append(calipso['lat'][i])
                colCal.lon0.append(calipso['lon'][i])

                if not firstLayerFound:
                    colCal.cod.append(calipso['colCOD'][i])
                    colCal.phase0.append(np.nan)
                    colCal.type0.append(np.nan)
                    colCal.ctp0Top.append(np.nan)
                    colCal.ctp0Bot.append(np.nan)
                    colCal.cth0Top.append(np.nan)
                    colCal.cth0Bot.append(np.nan)
                    colCal.ctt0Top.append(np.nan)

                if not secondLayerFound:
                    colCal.phase1.append(np.nan)
                    colCal.type1.append(np.nan)
                    colCal.ctp1Top.append(np.nan)
                    colCal.ctp1Bot.append(np.nan)
                    colCal.cth1Top.append(np.nan)
                    colCal.cth1Bot.append(np.nan)
                    colCal.ctt1Top.append(np.nan)
                    colCal.lat1.append(calipso['lat'][i])

                if not thirdLayerFound:
                    colCal.phase2.append(np.nan)
                    colCal.type2.append(np.nan)
                    colCal.ctp2Top.append(np.nan)
                    colCal.ctp2Bot.append(np.nan)
                    colCal.cth2Top.append(np.nan)
                    colCal.cth2Bot.append(np.nan)
                    colCal.ctt2Top.append(np.nan)
                    colCal.lat2.append(calipso['lat'][i])

    colCal.lists_to_arrays()
    colImg.lists_to_arrays()

    colCal.N = colCal.lat0.shape[0]
    colImg.N = colCal.lat0.shape[0]

    return colCal, colImg


def _plot_topo(colCal, ax, X):
    """ Plot topography and surface type in crosssection along track plot. """

    ncols = colCal.N
    colors = []

    for i in range(ncols):
        # if sfcTyp is not water or elevation > 0 use land color (green)
        if colCal.sfcTyp[i] != 17: # or colCal.sfcTop[i] > 0:
            colors.append('mediumseagreen')
        # else use blue for water bodies
        else:
            colors.append('blue')

    # boolean array having True if land surface
    excl = np.array([True if c == 'mediumseagreen' else False for c in colors])

    # to plot water: elevation array with land surfaces masked
    YY = np.where(excl, np.nan, colCal.sfcTop)

    # plot water
    ax.plot(YY / 1000, color='blue', linewidth=3)
    # plot land elevation
    ax.fill_between(X, colCal.sfcTop/1000,
                    color='mediumseagreen',
                    where=excl,
                    linewidth=3)


def _make_table(calipso, imager):
    """ Make cloud phase table. """

    # correct phases for plotting
    cphCal0 = _correct_calipso_phase(calipso.phase0)
    cphCal1 = _correct_calipso_phase(calipso.phase1)
    cphCal2 = _correct_calipso_phase(calipso.phase2)
    cphImg = _correct_cci_phase(imager.cph)

    df = DataFrame([cphCal0, cphCal1, cphCal2, cphImg])

    # generate table
    vals = np.around(df.values, 2)
    normal = plt.Normalize(np.nanmin(vals) - 1, np.nanmax(vals) + 1)
    cell_colours = plt.cm.RdBu(normal(vals))
    """cell colours scale from blue (phase = 1) to red (phase = 2)
        i.e. green is 0, and only red and blue depend on phase:
        red  = phase - 1.
        blue = 1. - red"""
    cell_colours[:, :, 0] = vals - 1.  # red
    cell_colours[:, :, 1] = 0.  # green
    cell_colours[:, :, 2] = 2. - vals  # blue
    cell_colours[np.isnan(vals), 0:3] = 1.
    cell_colours[abs(cell_colours) > 900.] = 0.5
    cell_colours[cell_colours[:, :, 0] == 0.5, 1] = 0.5
    cell_colours[:, :, 3] = 0.8

    row_labels = ["Calipso [COT > 0]", "Calipso [COT > 0.15]",
                  "Calipso [COT > 1]", "SEVIRI"]
    cell_text = np.chararray((len(row_labels), len(calipso.lat0)))

    cell_text[0,] = ''
    cell_text[1,] = ''
    cell_text[2,] = ''
    cell_text[3,] = ''

    table = plt.table(cellText=cell_text,
                      rowLabels=row_labels,
                      cellColours=cell_colours,
                      bbox=[0., -0.7, 1., 0.32],
                      loc='bottom')

    table.scale(2., 2.)

    # iterate through cells of a table
    table_props = table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell._text.set_color('white')
        cell.set_linewidth(0.2)
        cell.set_edgecolor('white')
    table._cells[(0, -1)]._text.set_color('black')
    table._cells[(1, -1)]._text.set_color('black')
    table._cells[(2, -1)]._text.set_color('black')
    table._cells[(3, -1)]._text.set_color('black')
    table._cells[(0, -1)].set_edgecolor('black')
    table._cells[(1, -1)].set_edgecolor('black')
    table._cells[(2, -1)].set_edgecolor('black')
    table._cells[(3, -1)].set_edgecolor('black')


def _along_track_plot(colCal, colImg, date, make_table, savepath):
    """ Make cross section plot along the collocation track. """

    if make_table:
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(2, 1, 1)
    else:
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 1, 1)

    ax.set_ylim([0, 30])
    ax.set_ylabel("height [km]")
    ax.set_xlabel('lat/lon/time')
    ax.set_facecolor('lightskyblue')
    ax.set_title(date + ' | # = {}'.format(colCal.N), fontweight='bold')

    # make xlabels (lat/lon/time)
    times = [datetime.fromtimestamp(i).strftime("%H:%M") for i in list(colCal.time)]
    xlabels = ['{:.1f}\n{:.1f}\n{}'.format(lat, lon, sec) for lat, lon, sec in
               zip(colCal.lat0, colCal.lon0, times)]

    # set xtick increment so that for every length 10 ticks are plotted
    nticks_wanted = 10
    if colCal.N > nticks_wanted:
        increment = colCal.N // nticks_wanted
    else:
        increment = 1

    # set xticks
    xx = np.arange(colCal.N)
    ax.set_xticklabels(xlabels[::increment])
    ax.set_xticks(xx[::increment])
    ax.set_xlim(xx.min(), xx.max())

    # calculate bar width and bar x positions
    width = (xx.max() - xx.min()) / xx.shape[0]
    X = xx - width * 0.5

    # plot CALIOP profiles
    alpha = 1
    ax.bar(X, colCal.cth0Top - colCal.cth0Bot,
           bottom=colCal.cth0Bot,
           width=width,
           color="lavenderblush",
           alpha=alpha,
           label="Calipso [COT > 0]",
           zorder=3,
           linewidth=0.5)
    ax.bar(X, colCal.cth1Top - colCal.cth1Bot,
           bottom=colCal.cth1Bot,
           width=width,
           color="thistle",
           alpha=alpha,
           label="Calipso [COT > 0.15]",
           zorder=3,
           linewidth=0.5)
    ax.bar(X, colCal.cth2Top - colCal.cth2Bot,
           bottom=colCal.cth2Bot,
           width=width,
           color="lightpink",
           alpha=alpha,
           label="Calipso [COT > 1]",
           zorder=3,
           linewidth=0.5)

    # plot topography + surface type
    _plot_topo(colCal, ax, X)

    # imager CTHs
    ax.scatter(xx, colImg.cth/1000,
               label="SEVIRI",
               c="black",
               zorder=10,
               marker='_',
               s=20)

    plt.legend()

    if make_table:
        _make_table(colCal, colImg)
    else:
        plt.tight_layout()

    if savepath is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(savepath)
        print('  ', savepath, 'saved ')
        plt.close()


def _seviri_grid(width, height):
    """
    Grid for SEVIRI geostationary CGMS projection. Before 6 Dec 2017 L1.5
    SEVIRI data were shifted by 1.5km SSP to North and West.
    This is the old grid before this date.
    """
    llx, lly, urx, ury = (-5570248.686685662,
                          -5567248.28340708,
                          5567248.28340708,
                          5570248.686685662)

    ogrid = pyresample.geometry.AreaDefinition(
        area_id='cgms',
        description='CGMS SEVIRI Grid',
        proj_id='geos',
        projection={'a': 6378169.0, 'b': 6356583.8, 'lon_0': 0.0,
                    'h': 35785831.0, 'proj': 'geos', 'units': 'm'},
        width=width,
        height=height,
        area_extent=(llx, lly, urx, ury)
    )

    return ogrid


def _rgb_track_plot(calipso, chans, datestring, savepath):
    """ Plot RGB of slot with collocation track. """

    # calculate rgb image
    rgb = calc_rgb_with016(chans['VIS006'],
                           chans['VIS008'],
                           chans['IR_016'],
                           chans['IR_108'],
                           chans['sunzen']).astype(np.uint8)

    # get geostationary AreaDefinition/CRS/extent
    geos = _seviri_grid(3712, 3712)
    iproj = geos.to_cartopy_crs()
    extent = iproj.bounds

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection=iproj)
    # plot RGB
    ax.imshow(np.fliplr(rgb), extent=extent)
    # plot collocation track
    ax.scatter(calipso['lon'], calipso['lat'],
               transform=ccrs.Geodetic(), s=1, color='red')
    # plot collocation start point thick
    ax.scatter(calipso['lon'][0], calipso['lat'][0],
               transform=ccrs.Geodetic(), s=60, color='red',
               marker='^', label='Start')
    # plot collocation end point thick
    ax.scatter(calipso['lon'][-1], calipso['lat'][-1],
               transform=ccrs.Geodetic(), s=60, color='red',
               marker='s', label='End')

    ax.coastlines(color='yellow', linewidth=.5)
    ax.set_title(datestring, fontweight='bold')

    plt.legend(loc=1)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
        print('  ', savepath, 'saved ')
    else:
        plt.show()

    plt.close()


def _get_measurements_from_hrit(hrit_files, year, month, day, slot):
    """ Read SEVIRI HRIT slot and calculate solar zenith angle for RGB. """
    scene = satpy.Scene(reader="seviri_l1b_hrit", filenames=hrit_files)
    scene.load(['VIS006', 'VIS008', 'IR_016', 'IR_108'])

    utc_time = datetime(int(year), int(month), int(day),
                        int(slot[:2]), int(slot[2:]))
    lon, lat = scene['VIS006'].attrs['area'].get_lonlats()
    sunzen = astronomy.sun_zenith_angle(utc_time, lon, lat)

    data = {'VIS006': scene['VIS006'].values,
            'VIS008': scene['VIS008'].values,
            'IR_016': scene['IR_016'].values,
            'IR_108': scene['IR_108'].values,
            'sunzen': sunzen,
            'lon': lon,
            'lat': lat}

    print('  READ HRIT DATA')

    return data, hrit_files


def plot_along_track(collocation_file, figpath=None, idx0=None, idx1=None,
                     make_table=True):
    """ Main function to run along track crossection plots. """

    # extract yyyymmdd and hhmm strings
    date = os.path.basename(collocation_file).split('_')[2]
    hhmm = os.path.basename(collocation_file).split('_')[3]

    # define datestring for plot title
    datestring = '{}/{}/{} {} UTC'.format(date[:4], date[4:6], date[6:], hhmm)

    # if figpath available define full path + output filename
    if figpath is not None:
        fig_ptf = os.path.join(figpath,
                               '{}_{}_alongtrack.png'.format(date, hhmm))
    else:
        fig_ptf = None

    # get data from atrain_match matchup files
    calipso, imager = _get_collocated_file_data(collocation_file, idx0, idx1)
    # prepare data for plotting
    colCal, colImg = _prepare_data(calipso, imager)

    print('  PROCESSING {}'.format(datestring))
    print('  LENGTH: {} COLLOCATIONS'.format(colCal.N))

    # do plotting
    _along_track_plot(colCal, colImg, datestring, make_table, fig_ptf)


def plot_rgb_track(collocation_file, hrit_files, date, slot, figpath=None,
                   idx0=None, idx1=None):
    """ Main function to run track overview RGB plots. """

    # define datestring for plot title
    datestring = '{}/{}/{} {} UTC'.format(date[:4], date[4:6], date[6:], slot)

    # if figpath available define full path + output filename
    if figpath is not None:
        fig_ptf = os.path.join(figpath,
                               '{}_{}_overview.png'.format(date, slot))
    else:
        fig_ptf = None

    # get data from atrain_match matchup files
    calipso, imager = _get_collocated_file_data(collocation_file, idx0, idx1)
    # read SEVIRI HRIT data for RGB
    chans, files = _get_measurements_from_hrit(hrit_files,
                                               date[:4],
                                               date[4:6],
                                               date[6:],
                                               slot)

    print('  PROCESSING {}'.format(datestring))
    print('  LENGTH: {} COLLOCATIONS'.format(calipso['lat'].shape[0]))

    # do plotting
    _rgb_track_plot(calipso, chans, datestring, fig_ptf)
