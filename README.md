# atrain_plot
Plot https://github.com/foua-pps/atrain_match collocated results on spatial maps and histograms.

Imports atrain_match scripts so atrain_match has to be available on your machine and should be appended to your PYTHONPATH environment variable.

Example:

#---------------------------

import atrain_plot

ipath_calipso='/path/to/your/hdf5/matchup/file/matchup_calipso_filename.h5'

opath='/path/where/figs/should/be/saved'

dnts = [ 'ALL', 'DAY', 'NIGHT', 'TWILIGHT' ]

satzs = [None, 70]

year = '2019'

month = '07'

dataset 'CCI' #CLAAS3

atrain_plot.run(ifilepath_calipso=ipath_calipso,  opath=opath, dnts=dnts, satzs=satzs, year=year, month=month, dataset=dataset)

#---------------------------
