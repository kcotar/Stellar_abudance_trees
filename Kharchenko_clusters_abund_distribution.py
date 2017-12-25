from os import chdir, path, remove
import imp
from glob import glob
from astropy.table import Table, vstack, join
import numpy as np
import matplotlib
matplotlib.rc('font', size=6)
import matplotlib.pyplot as plt

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import *
imp.load_source('helper2', '../tSNE_test/cannon3_functions.py')
from helper2 import *


def _prepare_hist_data(d, bins, range, norm=True):
    heights, edges = np.histogram(d, bins=bins, range=range)
    width = np.abs(edges[0] - edges[1])
    if norm:
        heights = 1.*heights / np.nanmax(heights)
    return edges[:-1], heights, width


data_dir = '/home/klemen/GALAH_data/'
cluster_dir = data_dir+'clusters/Kharchenko_2013/stars/'

date_string = '20171111'
cluster_fits = glob(cluster_dir+'*NGC_*'+date_string+'_galah_all.fits')

# galah_cannon_data = Table.read(data_dir + 'galah_cannon_3.0_ucac5_joined_'+date_string+'.fits')  # data with filtered abundances and known kinematics
galah_all = Table.read('/home/klemen/GALAH_data/' + 'sobject_iraf_52_reduced_20171111.fits')
galah_cannon_ucac5 = Table.read('/home/klemen/GALAH_data/' + 'galah_cannon_3.0_ucac5_joined_20171111.fits')

abund_cols = get_abundance_cols3(galah_cannon_ucac5.colnames)
abund_names = get_element_names(abund_cols)
n_abund = len(abund_names)

move_to_dir('Known_clusters_Kharchenko_2013')

for fits in cluster_fits:
    fits_file = fits.split('/')[-1]
    print fits_file
    fits_data = Table.read(fits)
    # filter out commissioning observations
    fits_data = fits_data[fits_data['sobject_id'] > 140301000000000]
    #
    if len(fits_data) > 0:
        fits_file_split = fits_file.split('_')
        cluster_name = fits_file_split[2] + '_' + fits_file_split[3]
        print cluster_name
        fits_data['cluster'] = cluster_name
        fits_data.sort('sobject_id')


        # most probable cluster members based on multiple parameters
        idx_probable = np.logical_and(np.logical_and(fits_data['Pkin'] > 70.0,
                                                     fits_data['PJH'] > 50.0),
                                      np.logical_and(fits_data['PJKs'] > 50.0,
                                                     fits_data['Ps'] >= 1))
        if np.sum(idx_probable) < 3:
            print ' Low number of probable members'
            continue

        n_probable = np.sum(idx_probable)
        print 'N prob (1): ', np.sum(idx_probable)
        cluster_galah_join = join(fits_data['sobject_id', 'cluster', 'Pkin', 'Ps'], galah_all['sobject_id', 'rv_guess'],
                                  keys='sobject_id')
        rv_mean = np.nanmedian(cluster_galah_join['rv_guess'][idx_probable])
        rv_std = np.nanstd(cluster_galah_join['rv_guess'][idx_probable])

        # select a subset based on the RV
        idx_probable = np.logical_and(np.logical_and(cluster_galah_join['Pkin'] >= 0.0,
                                                     np.abs(cluster_galah_join['rv_guess'] - rv_mean) < 0.3 * rv_std),
                                      fits_data['Ps'] >= 1)
        print 'N prob (2): ', np.sum(idx_probable)
        probable_members_ucac5 = galah_cannon_ucac5[np.in1d(galah_cannon_ucac5['sobject_id'], fits_data[idx_probable]['sobject_id'])]

        if len(probable_members_ucac5) <= 0:
            continue

        # perform proper motion filtering
        pmra_mean = np.nanmedian(probable_members_ucac5['pmra'])
        pmra_std = np.nanstd(probable_members_ucac5['pmra'])
        pmdec_mean = np.nanmedian(probable_members_ucac5['pmdec'])
        pmdec_std = np.nanstd(probable_members_ucac5['pmdec'])
        idx_pm = np.logical_and(np.abs(probable_members_ucac5['pmra'] - pmra_mean) < 0.5 * pmra_std,
                                np.abs(probable_members_ucac5['pmdec'] - pmdec_mean) < 0.5 * pmdec_std)

        # final subset
        idx_probable = np.in1d(fits_data['sobject_id'], probable_members_ucac5['sobject_id'][idx_pm])
        print 'N prob (3): ', np.sum(idx_probable)

        cluster_prob_sid = fits_data[idx_probable]['sobject_id']
        cluster_back_sid = fits_data[~idx_probable]['sobject_id']
        #
        cluster_prob = galah_cannon_ucac5[np.in1d(galah_cannon_ucac5['sobject_id'], cluster_prob_sid)]
        cluster_back = galah_cannon_ucac5[np.in1d(galah_cannon_ucac5['sobject_id'], cluster_back_sid)]
        print 'All:', len(fits_data), 'probable:', n_probable, 'valid abund:', len(cluster_prob)
        if len(cluster_prob) <= 0:
            continue

        # show abundance distribution for selected subset(s)

        # generate distribution plots independently for every chemical abundance
        n_abund = len(abund_cols)
        fig, ax = plt.subplots(5, 6)  # correction for more abundances in cannon3 data release
        # fig.set_size_inches()
        fig.suptitle(cluster_name+' all:{:.0f}  prob:{:.0f} valid:{:.0f}'.format(len(fits_data), n_probable, len(cluster_prob)))
        for i_a in range(n_abund):
            subplot_x = i_a % 6
            subplot_y = int(i_a / 6)
            # hist background
            h_edg, h_hei, h_wid = _prepare_hist_data(cluster_back[abund_cols[i_a]], 50, (-1.5, 0.8))
            ax[subplot_y, subplot_x].bar(h_edg, h_hei, width=h_wid, color='black', alpha=0.25)
            # hist interesting data
            n_unflagged = np.sum(np.isfinite(cluster_prob[abund_cols[i_a]]))
            h_edg, h_hei, h_wid = _prepare_hist_data(cluster_prob[abund_cols[i_a]], 50, (-1.5, 0.8))
            ax[subplot_y, subplot_x].bar(h_edg, h_hei, width=h_wid, color='red', alpha=0.5)
            # make it nice
            ax[subplot_y, subplot_x].set(title=abund_names[i_a] + ' [' + str(n_unflagged) + ']')
            ax[subplot_y, subplot_x].grid(True, color='black', linestyle='dashed', linewidth=1, alpha=0.1)

        # histogram of unflagged abundances
        abund_ok_object = np.sum(np.isfinite(cluster_prob[abund_cols].to_pandas().values), axis=1)
        ax[4, 5].hist(abund_ok_object, bins=n_abund+1, range=(0, n_abund+1), align='left')

        # plt.tight_layout()
        # plt.subplots_adjust(hspace=0.3, wspace=0.2, left=0.05, bottom=0.05, right=0.95, top=0.92)
        # plt.show()

        plt.subplots_adjust(hspace=0.6, wspace=0.3, left=0.05, bottom=0.05, right=0.95, top=0.88)
        plt.savefig(cluster_name+'.png', dpi=550)
        plt.close()
