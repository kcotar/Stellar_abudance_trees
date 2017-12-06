from os import chdir, path, remove
from glob import glob
from astropy.table import Table, vstack
import numpy as np

data_dir = '/home/klemen/GALAH_data/clusters/Kharchenko_2013/stars/'

suffix = ''
date_string = '20171111'
cluster_fits = glob(data_dir+'*NGC_*'+date_string+'_galah'+suffix+'.fits')

# NJ settings
radius = 10

txt_out = open('terminal_run_clusters.txt', 'w')
data = list([])
for fits in cluster_fits:
    fits_file = fits.split('/')[-1]
    print fits_file
    fits_data = Table.read(fits)
    if len(fits_data) > 0:
        fits_file_split = fits_file.split('_')
        cluster_name = fits_file_split[2]+'_'+fits_file_split[3]
        print cluster_name
        # add cluster name column to the data
        fits_data['cluster'] = cluster_name

        # append, print
        data.append(fits_data)
        ra_mean = np.mean(fits_data['ra'])
        dec_mean = np.mean(fits_data['dec'])
        print len(fits_data), ra_mean, dec_mean

        # create and save execution command
        command = 'nohup python NJ_tree_analysis_cannon_new.py'
        command += ' {:.1f} {:.1f} {:.0f}'.format(ra_mean, dec_mean, radius)
        command += ' --loose=True --join=True --metric=cityblocknans --method=weighted > /dev/null & \n'
        txt_out.write(command)

# close output file
txt_out.close()

# stack all read data together
chdir('/home/klemen/GALAH_data/clusters/')
out_file = '2m_all_'+date_string+'_galah'+suffix+'.fits'
data_all = vstack(data)
if path.isfile(out_file):
    remove(out_file)
data_all.write(out_file)
