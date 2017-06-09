import os, imp

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir

galah_data_dir = '/home/klemen/GALAH_data/'
# Galah parameters and abundance data
print 'Reading data'
# known stars in clusters in this dataset
stars_cluster_data = Table.read(galah_data_dir+'sobject_clusterstars_1.0.fits')
actions_data = Table.read(galah_data_dir+'galah_tgas_xmatc_actions.fits')

move_to_dir('Actions_clusters')

for cluster in np.unique(stars_cluster_data['cluster_name']):
    print cluster
    cluster_ids = stars_cluster_data['sobject_id'][stars_cluster_data['cluster_name']==cluster]
    idx_tgas = np.in1d(actions_data['sobject_id'], cluster_ids)
    n_in = np.sum(idx_tgas)
    print 'Objects:' +str(n_in)
    if n_in <= 0:
        continue
    for action in ['J_R', 'J_Z', 'L_Z']:
        a_data = actions_data[action]._data
        x_min = np.nanpercentile(a_data, 2)
        x_max = np.nanpercentile(a_data, 98)
        plt.hist(a_data, range=(x_min,x_max), bins=150, color='black', alpha=0.75, normed=True)
        plt.hist(a_data[idx_tgas], range=(x_min, x_max), bins=75, color='blue', alpha=0.75, normed=True)
        plt.savefig(cluster+'_'+action+'.png', dpi=250)
        plt.close()



