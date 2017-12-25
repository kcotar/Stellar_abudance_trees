from os import chdir, path, remove
from glob import glob
from astropy.table import Table, vstack, join
import numpy as np
from NJ_tree_analysis_functions import start_gui_explorer
import matplotlib.pyplot as plt


def output_plot(s_id, data, path=None, rv_range=None):
    data_sub = data[np.in1d(data['sobject_id'], s_id)]
    if len(data_sub) > 0:
        fig, ax = plt.subplots(1, 3, figsize=(15, 8))
        ax[0].hist(data_sub['rv_guess'], bins=50, range=(-100, 100))
        if rv_range is not None:
            ax[0].axvspan(rv_range[0], rv_range[1], color='black', alpha=0.3)
        ax[1].hist(data_sub['pmra'], bins=50, range=(-45., 45.))
        ax[2].hist(data_sub['pmdec'], bins=50, range=(-45., 45.))
        ax[0].set(xlabel='RV')
        ax[1].set(xlabel='pmra')
        ax[2].set(xlabel='pmdec')
        ax[0].grid(True, color='black', linestyle='dashed', linewidth=1, alpha=0.2)
        ax[1].grid(True, color='black', linestyle='dashed', linewidth=1, alpha=0.2)
        ax[2].grid(True, color='black', linestyle='dashed', linewidth=1, alpha=0.2)
        # plt.tight_layout()
        # plt.show()
        plt.savefig(path, dpi=250)
        plt.close()

cluster_dir = '/home/klemen/data4_mount/clusters/'

suffix = '_probable'
date_string = '20171111'
cluster_data_all = Table.read(cluster_dir+'2m_all_clusters_20171111_xmatch.fits')
galah_all = Table.read('/home/klemen/data4_mount/' + 'sobject_iraf_52_reduced_20171111.fits')
galah_cannon_ucac5 = Table.read('/home/klemen/data4_mount/' + 'galah_cannon_3.0_ucac5_joined_20171111.fits')

# NJ settings
radius = 7.

txt_out = open('terminal_run_clusters'+suffix+'.txt', 'w')
txt_out2 = open('terminal_run_clusters_sobject_ids'+suffix+'.txt', 'w')
data = list([])

chdir('Kharchenko_selection_final')

possible_clusters = np.unique(cluster_data_all['cluster'])
for cluster_name in possible_clusters:
    # fits_file = fits.split('/')[-1]
    # print fits_file
    # fits_data = Table.read(fits)
    fits_data = cluster_data_all[cluster_data_all['cluster'] == cluster_name]
    # filter out commissioning observations
    fits_data = fits_data[fits_data['sobject_id'] > 140301000000000]
    #
    if len(fits_data) > 0:
        # fits_file_split = fits_file.split('_')
        # cluster_name = fits_file_split[2]+'_'+fits_file_split[3]
        print cluster_name + ' -', len(fits_data)
        # add cluster name column to the data
        fits_data['cluster'] = cluster_name
        fits_data.sort('sobject_id')

        # most probable cluster members based on multiple parameters
        idx_probable = np.logical_and(np.logical_and(fits_data['Pkin'] > 30.0,
                                                     fits_data['PJH'] > 50.0),
                                      np.logical_and(fits_data['PJKs'] > 50.0,
                                                     fits_data['Ps'] >= 1))
        print 'N prob (1): ', np.sum(idx_probable)
        if np.sum(idx_probable) < 5:
            print ' Low number of probable members'
            continue

        # objs = [str(o) for o in fits_data['sobject_id'][idx_probable]]
        # start_gui_explorer(objs, manual=True, initial_only=False, loose=True,
        #                    kinematics_source='ucac5')

        cluster_galah_join = join(fits_data['sobject_id', 'cluster', 'Pkin', 'Ps'][idx_probable], galah_all['sobject_id', 'rv_guess', 'ra', 'dec'], keys='sobject_id')
        rv_mean = np.nanmedian(cluster_galah_join['rv_guess'])
        rv_std = np.nanstd(cluster_galah_join['rv_guess'])

        output_plot(cluster_galah_join['sobject_id'], galah_cannon_ucac5,
                    path=cluster_name + '_1.png', rv_range=(rv_mean-rv_std, rv_mean+rv_std))

        fits_data = cluster_galah_join[np.abs(cluster_galah_join['rv_guess'] - rv_mean) < rv_std]

        # # select a subset based on the RV
        # print rv_mean, '+/-', rv_std
        # idx_probable = np.logical_and(np.logical_and(cluster_galah_join['Pkin'] >= 0.0,
        #                                              np.abs(cluster_galah_join['rv_guess'] - rv_mean) < 0.3 * rv_std),
        #                               fits_data['Ps'] >= 1)
        #
        # print 'N prob (2): ', np.sum(idx_probable)
        # probable_members_ucac5 = galah_cannon_ucac5[np.in1d(galah_cannon_ucac5['sobject_id'], fits_data[idx_probable]['sobject_id'])]
        #
        # if len(probable_members_ucac5) < 3:
        #     continue
        #
        # output_plot(probable_members_ucac5['sobject_id'], galah_cannon_ucac5, cluster_name + '_2.png')
        #
        # # objs = [str(o) for o in probable_members_ucac5['sobject_id']]
        # # start_gui_explorer(objs, manual=True, initial_only=False, loose=True,
        # #                    kinematics_source='ucac5')
        #
        # # perform proper motion filtering
        # pmra_mean = np.nanmedian(probable_members_ucac5['pmra'])
        # pmra_std = np.nanstd(probable_members_ucac5['pmra'])
        # pmdec_mean = np.nanmedian(probable_members_ucac5['pmdec'])
        # pmdec_std = np.nanstd(probable_members_ucac5['pmdec'])
        # idx_pm = np.logical_and(np.abs(probable_members_ucac5['pmra'] - pmra_mean) < 0.4 * pmra_std,
        #                         np.abs(probable_members_ucac5['pmdec'] - pmdec_mean) < 0.4 * pmdec_std)
        #
        # # final subset
        # fits_data = fits_data[np.in1d(fits_data['sobject_id'], probable_members_ucac5['sobject_id'][idx_pm])]
        # print 'N prob (3): ', len(fits_data)
        #
        # output_plot(fits_data['sobject_id'], galah_cannon_ucac5, cluster_name + '_3.png')

        # append, print
        data.append(fits_data)
        ra_mean = np.mean(fits_data['ra'])
        dec_mean = np.mean(fits_data['dec'])
        print '', len(fits_data), ra_mean, dec_mean

        # create and save execution command
        txt_out.write(cluster_name+' {:.0f} \n'.format(len(fits_data)))
        command = 'nohup python NJ_tree_analysis_cannon_new.py'
        command += ' {:.1f} {:.1f} {:.0f}'.format(ra_mean, dec_mean, radius)
        command += ' --loose=True --join=True --metric=cityblocknans --method=ward --tsne=True > /dev/null & \n'
        txt_out.write(command)

        # output sobject_ids matched with possible stars
        txt_out2.write(cluster_name+' N:{:.0f}, coord: {:.1f} {:.1f} \n'.format(len(fits_data), ra_mean, dec_mean))
        txt_out2.write(','.join([str(s) for s in fits_data['sobject_id']])+'\n')
        txt_out2.write('\n')

        # objs = [str(o) for o in fits_data['sobject_id']]
        # start_gui_explorer(objs,
        #                    manual=True, initial_only=False, loose=True,
        #                    kinematics_source='ucac5')

# close output file
txt_out.close()
txt_out2.close()

# stack all read data together
chdir('/home/klemen/data4_mount/clusters/')
out_file = '2m_all_clusters_20171111_xmatch_probable.fits'
data_all = vstack(data)
if path.isfile(out_file):
    remove(out_file)
data_all.write(out_file)
