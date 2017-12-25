import os
from astropy.table import Table, vstack
from glob import glob
import numpy as np
from astropy import units as un
import astropy.coordinates as coord

data_dir = '/home/klemen/data4_mount/'
kharchenko_subdir = 'clusters/Kharchenko_2013/stars/'

fits_files = glob(data_dir + kharchenko_subdir + '2m_*.fits')
print 'Number of fits files:', len(fits_files)

date_string = '20171111'
galah_stars_pos = Table.read(data_dir + 'sobject_iraf_52_reduced_'+date_string+'.fits')
galah_stars_pos = galah_stars_pos[np.logical_and(galah_stars_pos['red_flag'] & 64 != 64,
                                                 galah_stars_pos['flag_guess'] == 0)]
galah_coords = coord.SkyCoord(ra=galah_stars_pos['ra'] * un.deg, dec=galah_stars_pos['dec'] * un.deg)

fits_all = list([])
for cluster_fits in fits_files:
    filename = cluster_fits.split('/')[-1][:-5]
    print 'Working on cluster file:', filename
    cluster_pos = Table.read(cluster_fits)
    ra_cluster_center = np.mean(cluster_pos['ra'])
    dec_cluster_center = np.mean(cluster_pos['dec'])
    cluster_center_coord = coord.SkyCoord(ra=ra_cluster_center * un.deg, dec=dec_cluster_center * un.deg)
    #
    galah_coords_sep = galah_coords.separation(cluster_center_coord)
    idx_sub = galah_coords_sep < (3. * un.deg)
    #
    if np.sum(idx_sub) <= 0:
        print ' Not observed'
        continue
    fits_all.append(cluster_pos)

os.chdir(data_dir + kharchenko_subdir)
os.chdir('..')
os.chdir('..')

fits_all = vstack(fits_all)
fits_all = fits_all[np.logical_and(fits_all['ra'] > 0., fits_all['ra'] < 360.)]

# add unique id that can be traced later
n_rows_final = len(fits_all)
fits_all['khar_id'] = np.int64(np.linspace(1, n_rows_final, n_rows_final))
fits_all.write('2m_all_clusters_'+date_string+'.fits', overwrite=True)

# save ra, dec and id only
fits_all['khar_id', 'ra', 'dec'].write('2m_all_clusters_pos_'+date_string+'.fits', overwrite=True)
