import os
import numpy as np

from astropy.table import Table, vstack

clusters_dir = '/home/klemen/GALAH_data/clusters/'

# ------------------------------------------------------------------
# -------------------- PART 1 :-------------------------------------
# -------------------- Combine sobject_ids of know cluster stars ---
# -------------------- (x-matched with different publications) -----
# ------------------------------------------------------------------

final_cluster_col = 'cluster'

# cluster members dataset 1
cluster_data_1 = Table.read(clusters_dir+'sobject_clusterstars_1.0.fits')
cluster_col = 'cluster_name'
cluster_data_1[final_cluster_col] = [str(n) for n in cluster_data_1[cluster_col]]
cluster_data_1['source'] = 'clusterstars'

# cluster members dataset 2
cluster_col = 'MWSC'
cluster_data_2 = Table.read(clusters_dir+'galah_clusters_Schmeja_xmatch_2014.csv', format='ascii.csv')
idx_probable = np.logical_and(np.logical_and(cluster_data_2['Pkin'] > 0.0,
                                             cluster_data_2['PJH'] > 0.0),
                              cluster_data_2['Ps'] == 1)
cluster_data_2 = cluster_data_2[idx_probable]
cluster_data_2[final_cluster_col] = [str(n) for n in cluster_data_2[cluster_col]]
cluster_data_2['source'] = 'Schmeja_2014'


# cluster members dataset 3
cluster_data_3 = Table.read(clusters_dir+'galah_clusters_Kharachenko_xmatch_2005.csv', format='ascii.csv')
cluster_col = 'Cluster'
cluster_data_3[final_cluster_col] = [str(n) for n in cluster_data_3[cluster_col]]
cluster_data_3['source'] = 'Kharachenko_2005'

# cluster members dataset 4
cluster_data_4 = Table.read(clusters_dir+'galah_clusters_Dias_xmatch_2014.csv', format='ascii.csv')
idx_probable = cluster_data_4['P'] > 50.0
idx_probable = np.logical_and(idx_probable, cluster_data_4['db'] == 0)
idx_probable = np.logical_and(idx_probable, cluster_data_4['of'] == 0)
cluster_data_4 = cluster_data_4[idx_probable]
cluster_col = 'Cluster'
cluster_data_4[final_cluster_col] = [str(n) for n in cluster_data_4[cluster_col]]
cluster_data_4['source'] = 'Dias_2014'

# join datasets
cluster_data_final = vstack([cluster_data_1['sobject_id', final_cluster_col, 'source'],
                             cluster_data_2['sobject_id', final_cluster_col, 'source'],
                             cluster_data_3['sobject_id', final_cluster_col, 'source'],
                             cluster_data_4['sobject_id', final_cluster_col, 'source']])

# save results
cluster_data_final_fits = clusters_dir+'galah_cluster_members_merged.fits'
if os.path.isfile(cluster_data_final_fits):
    os.remove(cluster_data_final_fits)
cluster_data_final.write(cluster_data_final_fits)


# ------------------------------------------------------------------
# -------------------- PART 2 :-------------------------------------
# -------------------- Prepare parameters of known clusters --------
# -------------------- (taken from different publications) ---------
# ------------------------------------------------------------------

# read the datasets

# clusters position and parameters dataset 1
cluster_centers_1 = Table.read(clusters_dir+'Dias_2014/table2.csv', format='ascii.csv')
cluster_centers_1['source'] = 'Dias_2014'

# clusters position and parameters dataset 2
cluster_centers_2 = Table.read(clusters_dir+'Kharchenko_2013/catalog.csv', format='ascii.csv')
cluster_centers_2['source'] = 'Kharchenko_2013'

# join datasets
cluster_centers_final = vstack([cluster_centers_1['Cluster', 'RAdeg', 'DEdeg', 'pmRAc', 'pmDEc', 'source'],
                                cluster_centers_2['Cluster', 'RAdeg', 'DEdeg', 'pmRAc', 'pmDEc', 'source']])

# save results
cluster_centers_final_fits = clusters_dir+'galah_cluster_parameters_merged.fits'
if os.path.isfile(cluster_centers_final_fits):
    os.remove(cluster_centers_final_fits)
cluster_centers_final.write(cluster_centers_final_fits)
