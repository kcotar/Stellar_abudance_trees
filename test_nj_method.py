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
from colorize_sky import *
from filter_galah import *

from ete3 import Tree

plot_abund_trees = True

galah_data_dir = '/home/klemen/GALAH_data/'
nj_data_dir = '/home/klemen/NJ_tree_settings/'
# Galah parameters and abundance data
galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon2.1.7.fits')
galah_tsne_class = Table.read(galah_data_dir+'tsne_class_1_0.csv')
galah_tgas_xmatch = Table.read(galah_data_dir+'galah_tgas_xmatch.csv')
galah_flats = Table.read(galah_data_dir+'flat_objects.csv')
# known stars in clusters in this dataset
stars_cluster_data = Table.read(galah_data_dir+'sobject_clusterstars_1.0.fits')
clusters_ra, clusters_dec = define_cluster_centers(stars_cluster_data, galah_cannon)
# get cannon abundance cols
abund_cols = get_abundance_cols(galah_cannon.colnames)
# determine which of those cols are ok to use (have at leas some data)
abund_cols_use = [col for col in abund_cols if np.isfinite(galah_cannon[col]).any()]

# create a filter that will be used to create a subset of data used in  abundance analysis
print 'Input data lines: '+str(len(galah_cannon))
print 'Creating filters for data subset'
galah_filter_ok = FILTER(verbose=True)
# remove data from the initial observations
galah_filter_ok.filter_attribute(galah_cannon, attribute='sobject_id', value=140203000000000, comparator='>')

# filter by cannon set flags - parameters flag, must be set to 0 for valid data
# galah_filter_ok._merge_ok(galah_cannon['flag_cannon'] == 0)

# filter by cannon set flags - abundances flags, all flags must be set to 0 for valid data
for abund_col in abund_cols_use:
    galah_filter_ok._merge_ok(galah_cannon['flag_'+abund_col] == 0)

# remove rows with any nan values in any of the abundance columns/attributes
galah_filter_ok.filter_valid_rows(galah_cannon, cols=abund_cols_use)

# filter by CCD SNR values
# galah_filter_ok.filter_attribute(galah_cannon, attribute='snr2_c1_iraf', value=20, comparator='>')

# filter by cannon CHI2 value
galah_filter_ok.filter_attribute(galah_cannon, attribute='chi2_cannon', value=90000, comparator='<')

# filter out problematic stars as detected by Gregor
galah_filter_ok.filter_objects(galah_cannon, galah_tsne_class, identifier='sobject_id')

# filter out flat target/object
galah_filter_ok.filter_objects(galah_cannon, galah_flats, identifier='sobject_id')

# filter out data that are not in tgas set
# galah_filter_ok.filter_objects(galah_cannon, galah_tgas_xmatch, identifier='sobject_id')

#  create a subset defined by filters above
galah_cannon_subset = galah_filter_ok.apply_filter(galah_cannon)

print 'Input filtered data lines: '+str(len(galah_cannon_subset))

# for ra_center, dec_center in zip([],[]):
#
# convert ra and dec to astropy coordinates
# ra_center = 57.
# dec_center = 24.
normalize = True
use_megacc = True
search_dist = 15.
ra_dec_coord = coord.ICRS(ra=np.array(galah_cannon_subset['ra'])*u.degree,
                          dec=np.array(galah_cannon_subset['dec'])*u.degree)

# normalize tho whole dataset at once
galah_cannon_subset_abund = galah_cannon_subset[abund_cols_use].to_pandas().values
if normalize:
    abund_norm_params = normalize_data(galah_cannon_subset_abund, method='standardize')

move_to_dir('NJ_tree_dwarfs_2.1.7_mainrun_abundflags_chi2_prob_clusters')
for i_grid in range(len(clusters_ra)):
    ra_center = clusters_ra[i_grid]
    dec_center = clusters_dec[i_grid]
    if ra_center is np.nan or dec_center is np.nan:
        continue
    stars_arc_dist = ra_dec_coord.separation(coord.ICRS(ra=ra_center*u.degree,
                                                        dec=dec_center*u.degree))
    stars_in_area = stars_arc_dist < search_dist*u.degree
    n_stars_in_area = np.sum(stars_in_area)
    print 'Number of stars found in search location: '+str(n_stars_in_area)
    if n_stars_in_area < 20 or n_stars_in_area > 20000:
        continue
    # create a subset of the dataset
    galah_cannon_cluster_abund = galah_cannon_subset_abund[stars_in_area]
    galah_cannon_cluster = galah_cannon_subset[stars_in_area]
    # galah_param_subset = galah_param[s

    move_to_dir('grid_{:03.0f}_ra_{:0.1f}_dec_{:0.1f}_rad_{:0.1f}'.format(i_grid, ra_center, dec_center, search_dist))
    output_nwm_file = 'distances_network.nwk'
    # compute distances and create phylogenetic tree from distance information
    if not os.path.isfile(output_nwm_file):
        distances = manhattan_distances(galah_cannon_cluster_abund)

        if use_megacc:
            # export distances to be used by megacc procedure
            txt = open('distances.meg', 'w')
            txt.write('#mega\n')
            txt.write('!Title Mega_distances_file;\n')
            txt.write('!Description Abundances_test_file;\n')
            txt.write('!Format DataType=distance DataFormat=lowerleft;\n')
            # export all taxa
            for s_id in galah_cannon_cluster['sobject_id']:
                txt.write('#'+str(s_id)+'\n')
            # output settings
            for i_r in range(1, distances.shape[0]):
                txt.write(' '.join(['{:0.4f}'.format(f) for f in distances[i_r, 0: i_r]])+'\n')
            txt.close()
            os.system('megacc -a '+nj_data_dir+'infer_NJ_distances.mao -d distances.meg -o '+output_nwm_file)
        else:
            # alternative way to build the tree, much slower option
            dm = DistanceMatrix(distances, galah_cannon_cluster['sobject_id'].data)
            nj_tree = nj(dm)
            dm = None
            # export tree to mwk file
            nj_tree.write(output_nwm_file, format='newick')
            # print(nj_tree.ascii_art())

    # read output tree file
    if plot_abund_trees:
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
        colorize_tree_branches(tree_struct, galah_cannon_cluster, 'logg_cannon', path='tree-branches_logg.png')
        colorize_tree_branches(tree_struct, galah_cannon_cluster, 'feh_cannon', path='tree-branches_feh.png')
        colorize_tree_branches(tree_struct, galah_cannon_cluster, 'teff_cannon', path='tree-branches_teff.png')
        colorize_tree(tree_struct, galah_cannon_cluster, 'logg_cannon', path='tree_logg.png')
        colorize_tree(tree_struct, galah_cannon_cluster, 'feh_cannon', path='tree_feh.png')
        colorize_tree(tree_struct, galah_cannon_cluster, 'teff_cannon', path='tree_teff.png')
        # colorize_tree(tree_struct, galah_param_subset, 'rv_guess', path='tree_rv.png')
        for abund in abund_cols_use:
            colorize_tree_branches(tree_struct, galah_cannon_cluster, abund, path='tree-branches_abund_'+abund+'.png')
            colorize_tree(tree_struct, galah_cannon_cluster, abund, path='tree_abund_'+abund+'.png')

    # traverse the tree from bottom to to the leaves of branches
    # plot_ra_dec_locations(galah_cannon_subset, path='sky_pos.png')
    # plot_ra_dec_attribute(galah_param_subset, 'rv_guess', path='sky_pos_rv.png')

    # return to parent directory
    os.chdir('..')