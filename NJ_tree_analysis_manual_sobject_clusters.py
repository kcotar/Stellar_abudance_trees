import imp
import numpy as np

from astropy.table import Table
from NJ_tree_analysis_functions import start_gui_explorer

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir

galah_dir = '/home/klemen/GALAH_data/'
out_dir = '/home/klemen/Stellar_abudance_trees/Known_clusters'
k_source = 'tgas'
cluster_id = 1

if cluster_id == 1:
    # clusters dataset 1
    cluster_data = Table.read(galah_dir+'sobject_clusterstars_1.0.fits')
    cluster_col = 'cluster_name'
    out_dir += '_clusterstars/'
elif cluster_id == 2:
    # clusters dataset 2
    cluster_data = Table.read(galah_dir+'galah_clusters_Schmeja_xmatch_2014.csv', format='ascii.csv')
    idx_probable = np.logical_and(np.logical_and(cluster_data['Pkin'] > 0.0, cluster_data['PJH'] > 0.0), cluster_data['Ps'] == 1)
    stars_cluster_data = cluster_data[idx_probable]
    cluster_col = 'MWSC'
    out_dir += '_Schmeja_2014/'
elif cluster_id == 3:
    # clusters dataset 3
    cluster_data = Table.read(galah_dir+'galah_clusters_Kharachenko_xmatch_2005.csv', format='ascii.csv')
    cluster_col = 'Cluster'
    out_dir += '_Kharachenko_2005/'
elif cluster_id == 4:
    # clusters dataset 4
    cluster_data = Table.read(galah_dir+'galah_clusters_Dias_xmatch_2014.csv', format='ascii.csv')
    idx_probable = cluster_data['P'] > 50.0
    idx_probable = np.logical_and(idx_probable, cluster_data['db'] == 0)
    idx_probable = np.logical_and(idx_probable, cluster_data['of'] == 0)
    stars_cluster_data = cluster_data[idx_probable]
    cluster_col = 'Cluster'
    out_dir += '_Dias_2014/'

# number of unique clusters
clusters = np.unique(cluster_data[cluster_col])
n_clust = len(clusters)
print 'Total observations: '+str(len(cluster_data))
print 'Total cluster: '+str(n_clust)

move_to_dir(out_dir)
for cur_cluster in clusters:
    objs = cluster_data[cluster_data[cluster_col] == cur_cluster]['sobject_id']

    objs = [str(o) for o in objs]
    start_gui_explorer(objs,
                       save_dir=out_dir+(str(cur_cluster).replace(' ',''))+'_'+k_source,
                       manual=False,
                       i_seq=None,
                       initial_only=True,
                       kinematics_source=k_source)
