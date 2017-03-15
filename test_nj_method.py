import os, imp
import numpy as np
import astropy.units as u
import astropy.coordinates as coord

from sklearn.metrics.pairwise import manhattan_distances
from skbio import DistanceMatrix
from skbio.tree import nj
from astropy.table import Table

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir, get_abundance_cols, define_cluster_centers
imp.load_source('norm', '../Stellar_parameters_interpolator/data_normalization.py')
from norm import *

from colorize_tree import *

from ete3 import Tree


galah_data_dir = '/home/klemen/GALAH_data/'
nj_data_dir = '/home/klemen/NJ_tree_settings/'
# Galah parameters and abundance data
galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')
# known stars in clusters in this dataset
stars_cluster_data = Table.read(galah_data_dir+'sobject_clusterstars_1.0.fits')
clusters_ra, clusters_dec = define_cluster_centers(stars_cluster_data, galah_cannon)
abund_cols = get_abundance_cols(galah_cannon.colnames)

# for ra_center, dec_center in zip([],[]):
#
# convert ra and dec to astropy coordinates
ra_center = 57.
dec_center = 24.
search_dist = 4.
ra_dec_coord = coord.ICRS(ra=np.array(galah_cannon['ra'])*u.degree,
                          dec=np.array(galah_cannon['dec'])*u.degree)
for ra_center, dec_center in zip(clusters_ra, clusters_dec):
    if ra_center is np.nan or dec_center is np.nan:
        continue
    stars_arc_dist = ra_dec_coord.separation(coord.ICRS(ra=ra_center*u.degree, dec=dec_center*u.degree))
    stars_in_area = stars_arc_dist < search_dist*u.degree
    n_stars_in_area = np.sum(stars_in_area)
    print 'Number of stars found in search location: '+str(n_stars_in_area)
    if n_stars_in_area < 50:
        continue
    # create a subset of the dataset
    galah_cannon_subset = galah_cannon[stars_in_area]
    galah_cannon_abund_data_sebset = np.array(galah_cannon_subset[abund_cols].to_pandas())
    norm_params = normalize_data(galah_cannon_abund_data_sebset, method='standardize')

    move_to_dir('Stellar_neighbour_tree_ra_{:0.1f}_dec_{:0.1f}_rad_{:0.1f}'.format(ra_center, dec_center, search_dist))
    output_nwm_file = 'distances_network.nwk'
    # compute distances and create phylogenetic tree from distance information
    if not os.path.isfile(output_nwm_file):
        distances = manhattan_distances(galah_cannon_abund_data_sebset)
        # export distances to be used by megacc procedure
        txt = open('distances.meg', 'w')
        txt.write('#mega\n')
        txt.write('!Title Mega_distances_file;\n')
        txt.write('!Description Abundances_test_file;\n')
        txt.write('!Format DataType=distance DataFormat=lowerleft;\n')
        # export all taxa
        for s_id in galah_cannon_subset['sobject_id']:
            txt.write('#'+str(s_id)+'\n')
        # output settings
        for i_r in range(1, distances.shape[0]):
            txt.write(' '.join(['{:0.4f}'.format(f) for f in distances[i_r, 0: i_r]])+'\n')
        txt.close()
        os.system('megacc -a '+nj_data_dir+'infer_NJ_distances.mao -d distances.meg -o '+output_nwm_file)

        # - OR -
        # alternative way to build the tree, much slower option
        # dm = DistanceMatrix(distances)
        # nj_tree = nj(dm)
        # print(nj_tree.ascii_art())

    # read output tree file
    if os.path.isfile(output_nwm_file):
        txt = open(output_nwm_file, 'r')
        tree_struct = Tree(file.readline(txt)[:-1])
        txt.close()
    else:
        print 'ERROR: check megacc as it did not produce the following file '+output_nwm_file
    print 'Plotting graphs'
    for cluster in np.unique(stars_cluster_data['cluster_name']):
        cluster_targets = stars_cluster_data[stars_cluster_data['cluster_name'] == cluster]['sobject_id']
        mark_objects(tree_struct, cluster_targets, path='cluster_'+cluster+'.png')
    colorize_tree(tree_struct, galah_cannon_subset, 'logg_cannon', path='tree_logg.png')
    colorize_tree(tree_struct, galah_cannon_subset, 'feh_cannon', path='tree_feh.png')
    colorize_tree(tree_struct, galah_cannon_subset, 'teff_cannon', path='tree_teff.png')
    for abund in abund_cols:
        colorize_tree(tree_struct, galah_cannon_subset, abund, path='tree_abund_'+abund+'.png')
    os.chdir('..')

