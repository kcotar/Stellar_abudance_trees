import os
import numpy as np

from astropy.table import Table
from astropy import units as un
import astropy.coordinates as coord
import pandas as pd
from glob import glob

def parse_kharchenko_cluster_dat(path):
    out_cols = ('ra','dec','B','V','J','H','Ks','e_J','e_H','e_Ks','pmra','pmdec','e_pm','RV','e_RV',
                'flags','2MASS','ASCC','SpType','Rcl','Ps','Pkin','PJKs','PJH','MWSC')
    spectype_pos = out_cols.index('SpType')
    # create a Table that will hold read data
    data_out = Table(names=out_cols,
                     dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                            'S15', 'i8', 'i8', 'S15', 'f8', 'i8', 'f8', 'f8', 'f8', 'i8'))
    # open txt dat file
    txt = open(path, 'r')
    txt_data = txt.readlines()
    txt.close()
    # iterate over lines
    n_cols = len(out_cols)
    for line in txt_data:
        row_values = filter(None, line[:-1].split(' '))
        n_vals = len(row_values)
        if n_vals < n_cols:
            # add one data that is missing
            row_values.insert(spectype_pos, '')
        elif n_vals > n_cols:
            # remove one data col
            for iii in range(n_vals - n_cols):
                row_values.pop(spectype_pos+1)
        data_out.add_row(row_values)
    data_out['ra'] *= 15.  # hour to degree
    return data_out


data_dir = '/home/klemen/GALAH_data/'
kharchenko_subdir = 'clusters/Kharchenko_2013/stars/'
date_string = '20171111'
galah_stars_pos = Table.read(data_dir + 'sobject_iraf_52_reduced_'+date_string+'.fits')
galah_stars_pos = galah_stars_pos[np.logical_and(galah_stars_pos['red_flag'] & 64 != 64,
                                                 galah_stars_pos['flag_guess'] == 0)]

# galah_stars_pos = galah_stars_pos[np.logical_and(galah_stars_pos['sobject_id']>170122002601000,
#                                                  galah_stars_pos['sobject_id']<170122002601400)]
print len(galah_stars_pos)

dat_files = glob(data_dir + kharchenko_subdir + '2m_*NGC_*.dat')
print 'Number of dat files:', len(dat_files)

for cluster_dat in dat_files[440:]:
    filename = cluster_dat.split('/')[-1][:-4]
    out_filename = data_dir + kharchenko_subdir + filename + '_'+date_string+'_galah_all.fits'
    print 'Working on cluster file:', filename
    print ' '+str(dat_files.index(cluster_dat)+1)+' of '+str(len(dat_files))
    if os.path.isfile(out_filename):
        print ' Already exists, skipping.'
        continue

    # read and parse dat file
    print ' Parsing dat file'
    cluster_pos = parse_kharchenko_cluster_dat(cluster_dat)
    # cluster_pos = Table.read(filename+'.fits')

    # most probable cluster members based on multiple parameters
    # idx_probable = np.logical_and(np.logical_and(cluster_pos['Pkin'] > 10.0,
    #                                              cluster_pos['PJH'] > 0.0),
    #                               np.logical_and(cluster_pos['PJKs'] > 0.,
    #                                              cluster_pos['Ps'] >= 0))
    # #
    # print ' Probable stars in cluster: {:.1f}%'.format(100.*np.sum(idx_probable)/len(idx_probable))
    #
    # # filter out interesting objects
    # cluster_pos = cluster_pos[idx_probable]

    # determine center pos of this cluster
    ra_cluster_center = np.mean(cluster_pos['ra'])
    dec_cluster_center = np.mean(cluster_pos['dec'])
    cluster_center_coord = coord.SkyCoord(ra=ra_cluster_center*un.deg, dec=dec_cluster_center*un.deg)
    print ' Center ra:{:.2f} dec:{:.2f}'.format(ra_cluster_center, dec_cluster_center)

    # matched stars - add new col to the input fits file
    cluster_pos['sobject_id'] = -1

    # manual xmatch - might be slow
    max_sep = 1.5 * un.arcsec

    # subset of galah data to be matched with a given list of stars in cluster
    galah_coords = coord.SkyCoord(ra=galah_stars_pos['ra'] * un.deg, dec=galah_stars_pos['dec'] * un.deg)
    galah_coords_sep = galah_coords.separation(cluster_center_coord)
    idx_sub = galah_coords_sep < (4.*un.deg)

    if np.sum(idx_sub) > 0:
        # for faster computation create a subset of galah data
        galah_coords_sub = galah_coords[idx_sub]
        galah_stars_pos_sub = galah_stars_pos[idx_sub]
        # perform data matching
        n_matches = 0
        n_possible = len(cluster_pos)
        for id_star in range(n_possible):
            if id_star % 250 == 0:
                print ' ', id_star+1, 'out of', n_possible
            star = cluster_pos[id_star]
            star_coord = coord.SkyCoord(ra=star['ra']*un.deg, dec=star['dec']*un.deg)
            idx_match = np.where(galah_coords_sub.separation(star_coord) <= max_sep)
            idx_match = idx_match[0]
            n_idx_match = len(idx_match)
            if n_idx_match > 0:
                print galah_stars_pos_sub[idx_match]['sobject_id'].data
                if n_idx_match == 1:
                    cluster_pos[id_star]['sobject_id'] = galah_stars_pos_sub[idx_match[0]]['sobject_id']
                else:
                    # print 'More than one match'
                    row_data = cluster_pos[id_star]
                    for s_sid_match in galah_stars_pos_sub[idx_match]['sobject_id']:
                        row_data['sobject_id'] = s_sid_match
                        cluster_pos.add_row(row_data)

        print ' Total matches:', n_matches
    else:
        print ' Fields not overlapping.'

    cluster_pos = cluster_pos[cluster_pos['sobject_id'] > 0]
    cluster_pos.write(out_filename)

