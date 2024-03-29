import imp, sys
import itertools

import numpy as np
import astropy.units as un
import astropy.coordinates as coord
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.preprocessing import StandardScaler, scale
from scipy.cluster.hierarchy import linkage, to_tree
from skbio import DistanceMatrix
from skbio.tree import nj
from astropy.table import Table, join
from getopt import getopt

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir, get_abundance_cols, define_cluster_centers, determine_clusters_abundance_deviation, get_multipler, string_bool
imp.load_source('norm', '../Stellar_parameters_interpolator/data_normalization.py')
from norm import *

imp.load_source('distf', '../tSNE_test/distances.py')
from distf import *
imp.load_source('esd_dist', '../tSNE_test/esd_distance.py')
from esd_dist import *

from colorize_tree import *
from filter_galah import *
from NJ_tree_analysis_functions import *

from ete3 import Tree


# ------------------------------------------------------------------
# -------------------- Functions -----------------------------------
# ------------------------------------------------------------------
# relocated for general use to the NJ_tree_analysis_functions.py

# ------------------------------------------------------------------
# -------------------- Program settings ----------------------------
# ------------------------------------------------------------------
# input arguments
input_arguments = sys.argv

# data settings
join_repeated_obs = True
normalize_abund = True
weights_abund = False
plot_overall_graphs = False
perform_data_analysis = True
investigate_repeated = True
save_results = True
manual_GUI_investigation = False
tgas_ucac5_use = 'ucac5'  # valid options are 'tgas', 'ucac5' and 'gaia'(when available)
use_cannon2 = False

# positional filtering settings
linkage_metric = 'esd'

loose_pairs = True
filter_by_sky_position = True
if len(input_arguments) > 3:
    print 'Using arguments given from command line'
    ra_center = float(input_arguments[1])
    dec_center = float(input_arguments[2])
    position_radius = float(input_arguments[3])
    # select only input arguments including -- string
    input_options = [arg for arg in input_arguments if '--' in arg]
    if len(input_options) > 0:
        opts, args = getopt(input_options, '', ['metric=', 'join=', 'loose='])
        # set parameters, depending on user inputs
        print input_arguments
        print opts
        for o, a in opts:
            if o == '--metric':
                linkage_metric = a
            if o == '--join':
                join_repeated_obs = string_bool(a)
            if o == '--loose':
                loose_pairs = string_bool(a)

else:
    ra_center = 30.  # degrees
    dec_center = 90.  # degrees
    position_radius = 35.  # degrees

# tree generation algorithm settings
use_megacc = False
phylip_shape = False
hierachical_scipy = True
# how to name the output dir
suffix = ''

# ------------------------------------------------------------------
# -------------------- Data reading and initial handling -----------
# ------------------------------------------------------------------

galah_data_dir = '/home/klemen/GALAH_data/'
actions_data_dir = '/home/klemen/Aquarius_membership/'
nj_data_dir = '/home/klemen/NJ_tree_settings/'
trees_dir = '/home/klemen/Stellar_abudance_trees/'
# Galah parameters and abundance data
print 'Reading data'
if use_cannon2:
    galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon2.1.7.fits')
    galah_cannon.remove_columns(['rv_guess', 'teff_guess', 'logg_guess', 'feh_guess'])
else:
    galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')
galah_cannon['galah_id'].name = 'galah_id_old'  # rename old reduction parameters
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.fits')['sobject_id', 'galah_id', 'rv_guess', 'teff_guess', 'logg_guess', 'feh_guess', 'snr_c1_iraf', 'snr_c1_guess', 'snr_c2_guess', 'snr_c3_guess', 'snr_c4_guess']
# reduction classification
galah_tsne_class = Table.read(galah_data_dir+'tsne_class_1_0.csv')
galah_flats = Table.read(galah_data_dir+'flat_objects.csv')
# additional kinematics data
if tgas_ucac5_use is 'tgas':
    galah_kinematics_xmatch = Table.read(galah_data_dir + 'galah_tgas_xmatch.csv')['sobject_id', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'parallax', 'parallax_error']
elif tgas_ucac5_use is 'ucac5':
    galah_kinematics_xmatch = Table.read(galah_data_dir + 'galah_ucac5_xmatch.csv')['sobject_id', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error']

# join datasets
galah_cannon = join(galah_cannon, galah_param, keys='sobject_id', join_type='inner')
# known stars in clusters in this dataset
stars_cluster_data = Table.read(galah_data_dir+'clusters/sobject_clusterstars_1.0.fits')
clusters_ra, clusters_dec = define_cluster_centers(stars_cluster_data, galah_cannon)

# get cannon abundance cols
abund_cols = get_abundance_cols(galah_cannon.colnames)
# determine which of those cols are ok to use (have at least some data)
abund_cols_use = [col for col in abund_cols if np.isfinite(galah_cannon[col]).any()]

# create a filter that will be used to create a subset of data used in  abundance analysis
print 'Input data lines: '+str(len(galah_cannon))
print 'Creating filters for data subset'
galah_filter_ok = FILTER(verbose=True)
# remove data from the initial observations
galah_filter_ok.filter_attribute(galah_cannon, attribute='sobject_id', value=140203000000000, comparator='>')

# ------------------------------------------------------------------
# -------------------- Abundance data filtering --------------------
# ------------------------------------------------------------------

if use_cannon2:
    # use for Cannon 2.1.7
    # filter by cannon set flags - abundances flags, all flags must be set to 0 for valid data
    galah_filter_ok._merge_ok(np.isfinite(galah_cannon[abund_cols_use].to_pandas().values).all(axis=1))
    for abund_col in abund_cols_use:
        galah_filter_ok._merge_ok(galah_cannon['flag_'+abund_col] == 0)
else:
    # use for Cannon 1.2
    # filter by cannon set flags - parameters flag, must be set to 0 for valid data
    galah_filter_ok._merge_ok(galah_cannon['flag_cannon'] == '0.0')

# remove rows with any nan values in any of the abundance columns/attributes
galah_filter_ok.filter_valid_rows(galah_cannon, cols=abund_cols_use)

# filter by CCD SNR values
# galah_filter_ok.filter_attribute(galah_cannon, attribute='snr2_c1_iraf', value=20, comparator='>')

# filter by cannon CHI2 value
# about 35-40k for 1.2 and 75k for 2.1.7
galah_filter_ok.filter_attribute(galah_cannon, attribute='chi2_cannon', value=35000, comparator='<')
galah_filter_ok.filter_attribute(galah_cannon, attribute='snr_c1_iraf', value=15, comparator='>')

# filter out classes of possibly problematic stars as detected by Gregor
# determine which problematic stars might be problematic for abundance determination
galah_filter_ok.filter_objects(galah_cannon, galah_tsne_class[galah_tsne_class['published_reduced_class_proj1']=='binary'],
                               identifier='sobject_id')
galah_filter_ok.filter_objects(galah_cannon, galah_tsne_class[galah_tsne_class['published_reduced_class_proj1']=='mol. absorption bands'],
                               identifier='sobject_id')
galah_filter_ok.filter_objects(galah_cannon, galah_tsne_class[np.logical_and(galah_tsne_class['published_reduced_class_proj1']=='problematic',
                                                                             ['emission' not in s for s in galah_tsne_class['unreduced_flag_proj1']])],
                               identifier='sobject_id')

# filter out flat target/object
# galah_filter_ok._merge_ok(np.logical_not(np.bitwise_and(galah_cannon['red_flags'], 64) == 64))
galah_filter_ok.filter_objects(galah_cannon, galah_flats, identifier='sobject_id')

# ------------------------------------------------------------------------
# statistics based on a dateset of good quality
_temp_sub = galah_filter_ok.apply_filter(galah_cannon)
if weights_abund:
    # at this point determine cluster deviations and normalize data, TGAS filtering is performed after that
    std_abundances = determine_clusters_abundance_deviation(_temp_sub, stars_cluster_data, abund_cols_use)
    print std_abundances
if normalize_abund:
    # determine normalization parameters
    # standardize (mean=0, std=1) individual abundace column
    norm_params_full = normalize_data(_temp_sub[abund_cols_use].to_pandas().values, method='standardize')
_temp_sub = None
# ------------------------------------------------------------------------

# filter out data that are not in kinematics dataset
suffix += '_'+tgas_ucac5_use
galah_filter_ok.filter_objects(galah_cannon, galah_kinematics_xmatch, identifier='sobject_id', invert=False)

# create a subset defined by filters above
galah_cannon_subset = galah_filter_ok.apply_filter(galah_cannon)


# ------------------------------------------------------------------
# -------------------- Join repeated observations ------------------
# ------------------------------------------------------------------
if join_repeated_obs:
    # TODO export repeated observations
    print 'Merging repeated observations'
    suffix += '_norep'
    compute_mean_cols = list(abund_cols_use)
    compute_mean_cols.append('rv_guess')
    i_rep, c_rep = np.unique(galah_cannon_subset['galah_id'], return_counts=True)
    ids_join = i_rep[np.logical_and(i_rep > 0, c_rep >= 2)]
    n_reps = np.sum(c_rep[np.logical_and(i_rep > 0, c_rep >= 2)])
    if len(ids_join) > 0:
        print 'Number of repeated object: '+str(len(ids_join))+' and observations: '+str(n_reps)
        for id_join in ids_join:
            idx_rows = np.where(galah_cannon_subset['galah_id'] == id_join)
            mean_cols = np.nanmedian(galah_cannon_subset[idx_rows][compute_mean_cols].to_pandas().values, axis=0)
            out_row = idx_rows[0][0]
            for i_col in range(len(compute_mean_cols)):
                galah_cannon_subset[out_row][compute_mean_cols[i_col]] = mean_cols[i_col]
            galah_cannon_subset.remove_rows(idx_rows[0][1:])
    print 'Object after joining repeated: '+str(len(galah_cannon_subset))

# remove problems with masks
# galah_cannon_subset = galah_cannon_subset.filled()

# ------------------------------------------------------------------
# -------------------- Positional filtering for large datasets -----
# ------------------------------------------------------------------
suffix_pos = ''
if filter_by_sky_position:
    print 'Filtering stars by their position on the sky'
    print 'RA center:', ra_center
    print 'DEC center:', dec_center
    print 'Radius center:', position_radius
    galah_cannon_subset['ra'].unit = ''
    galah_cannon_subset['dec'].unit = ''
    suffix_pos += '_ra_{:.1f}_dec_{:.1f}_rad_{:.1f}'.format(ra_center, dec_center, position_radius)
    galah_pos = coord.ICRS(ra=galah_cannon_subset['ra'] * un.deg,
                           dec=galah_cannon_subset['dec'] * un.deg)
    distance_to_center = galah_pos.separation(coord.ICRS(ra=ra_center * un.deg,
                                                         dec=dec_center * un.deg))
    idx_pos_select = distance_to_center <= position_radius*un.deg
    galah_cannon_subset = galah_cannon_subset[idx_pos_select]


# ------------------------------------------------------------------
# -------------------- Final filtering and data preparation --------
# ------------------------------------------------------------------

# galah_cannon_subset = galah_cannon_subset[:5]
# subset to numpy array of abundance values
galah_cannon_subset_abund = galah_cannon_subset[abund_cols_use].to_pandas().values
# standardize (mean=0, std=1) individual abundace column
if normalize_abund:
    suffix += '_norm'
    print 'Normalizing abundances'
    norm_params = normalize_data(galah_cannon_subset_abund, method='standardize', norm_param=norm_params_full)
# apply weights to the normalized abund parameters if
if weights_abund:
    suffix += '_weight'
    print 'Abundance weighting'
    for i_col in range(len(abund_cols_use)):
        ab_multi = get_multipler(std_abundances[i_col])
        print ' ', abund_cols_use[i_col], ab_multi, std_abundances[i_col]
        galah_cannon_subset_abund[:, i_col] *= ab_multi

print 'Input filtered data lines: '+str(len(galah_cannon_subset))

if phylip_shape:
    output_nwm_file = 'outtree'
    suffix += '_phylip'
elif use_megacc:
    output_nwm_file = 'distances_network.nwk'
    suffix += '_megacc'
    # distance computation
    suffix += '_manhattan'
elif hierachical_scipy:
    output_nwm_file = 'distances_network.nwk'
    suffix += '_hier'
    # method
    suffix += '_weighted'
    # distance computation
    suffix += '_'+linkage_metric
if loose_pairs:
    suffix += '_loose'

final_dir = 'NJ_tree'
if use_cannon2:
    final_dir += '_cannon_2.1.7'
else:
    final_dir += '_cannon_1.2'

final_dir += '_mainrun_abundflags_chi2_prob'+suffix+suffix_pos
# final_dir = 'NJ_tree_cannon_1.2_mainrun_abundflags_chi2_prob_tgas_norep_norm_megacc_manhattan'
move_to_dir(final_dir)

if filter_by_sky_position:
    # create an image of investigated sky area
    print 'Plotting sky area'
    grid_ra = np.linspace(0, 360, 360 * 4)
    grid_dec = np.linspace(-90, 90, 180 * 4)
    _dec, _ra = np.meshgrid(grid_dec, grid_ra)
    loc_observed = coord.ICRS(ra=_ra * un.deg,
                              dec=_dec * un.deg).separation(coord.ICRS(ra=ra_center * un.deg,
                                                                       dec=dec_center * un.deg)) <= position_radius * un.deg
    observed_field = np.int8(loc_observed).reshape(_dec.shape)
    fig, ax = plt.subplots(1, 1)
    im_ax = ax.imshow(observed_field.T, interpolation=None, cmap='seismic', origin='lower', vmin=0, vmax=1)
    fig.colorbar(im_ax)
    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig('area_sky_observed.png', dpi=300)
    plt.close()

# ------------------------------------------------------------------
# -------------------- Tree computation ----------------------------
# ------------------------------------------------------------------

# compute distances and create phylogenetic tree from distance information
if not os.path.isfile(output_nwm_file):
    print 'Computing data distances'

    if phylip_shape or use_megacc:
        # distances might use too much RAM, especially for more complex matrix distance computations
        distances = np.abs(manhattan_distances(galah_cannon_subset_abund))
        print distances[:5, :5]  # just a quick sanity check
        # OR any other distance computation from distances.py
        # manhattan_distances, euclidean_distances
        # canberra_distance, kulczynski_distance, sorensen_distance, bray_curtis_similarity, czekanovski_dice_distance

    if phylip_shape:
        # export distances to be used by megacc procedure
        txt = open('dist_phylip.txt', 'w')
        txt.write(str(distances.shape[0])+'\n')
        # export all taxa
        print 'Exporting taxa'
        # output settings
        time_start = time.time()
        for i_r in range(0, distances.shape[0]):
            if i_r % 2000 == 0:
                minutes = (time.time() - time_start) / 60.
                print ' {0}: {1} min'.format(i_r, minutes)
                time_start = time.time()
            txt.write(str(galah_cannon_subset[i_r]['sobject_id'])+'     '+' '.join(['{:0.4f}'.format(f) for f in distances[i_r, 0: i_r+1]]) + '\n')
        txt.close()
        print 'Now manually run phylip distance module and rerun this script'
        raise SystemExit
    elif use_megacc:
        # export distances to be used by megacc procedure
        txt = open('distances.meg', 'w')
        txt.write('#mega\n')
        txt.write('!Title Mega_distances_file;\n')
        txt.write('!Description Abundances_test_file;\n')
        txt.write('!Format DataType=distance DataFormat=lowerleft;\n')
        # export all taxa
        print 'Exporting taxa'
        for s_id in galah_cannon_subset['sobject_id']:
            txt.write('#'+str(s_id)+'\n')
        # output settings
        time_start = time.time()
        for i_r in range(1, distances.shape[0]):
            if i_r % 2000 == 0:
                minutes = (time.time() - time_start) / 60.
                print ' {0}: {1} min'.format(i_r, minutes)
                time_start = time.time()
            txt.write(' '.join(['{:0.4f}'.format(f) for f in distances[i_r, 0: i_r]])+'\n')
        txt.close()
        print 'Running megacc software'
        os.system('megacc -a '+nj_data_dir+'infer_NJ_distances.mao -d distances.meg -o '+output_nwm_file)
    elif hierachical_scipy:
        print 'Hierarchical clustering started'
        if linkage_metric is 'esd':
            linkage_matrix = linkage(esd_dist_compute(galah_cannon_subset_abund, triu=True), method='weighted')
        else:
            linkage_matrix = linkage(galah_cannon_subset_abund, method='weighted', metric=linkage_metric)  # might use too much RAM
        linkage_tree = to_tree(linkage_matrix, False)
        newic_tree_str = getNewick(linkage_tree, "", linkage_tree.dist, galah_cannon_subset['sobject_id'].data)
        nwm_txt = open(output_nwm_file, 'w')
        nwm_txt.write(newic_tree_str)
        nwm_txt.close()
    else:
        # alternative way to build the tree, much slower option
        print 'Initialize distance matrix object'
        dm = DistanceMatrix(distances, galah_cannon_subset['sobject_id'].data)  # data, ids
        print 'Generating tree from distance matrix'
        nj_tree = nj(dm)
        dm = None
        # export tree to mwk file
        nj_tree.write(output_nwm_file, format='newick')
        # print(nj_tree.ascii_art())

# read output tree file
tree_struct = get_tree_from_file(output_nwm_file)

if plot_overall_graphs:
    print 'Plotting graphs'
    # for cluster in np.unique(stars_cluster_data['cluster_name']):
    #     print cluster
    #     cluster_targets = stars_cluster_data[stars_cluster_data['cluster_name'] == cluster]['sobject_id']
    #     mark_objects(tree_struct, cluster_targets, path='cluster_'+cluster+'.png')
    # colorize_tree(tree_struct, galah_cannon_subset, 'logg_cannon', path='tree_logg.png')
    # colorize_tree(tree_struct, galah_cannon_subset, 'feh_cannon', path='tree_feh.png')
    # colorize_tree(tree_struct, galah_cannon_subset, 'teff_cannon', path='tree_teff.png')
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'logg_cannon', path='tree_logg_branches.png')
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'feh_cannon', path='tree_feh_branches.png')
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'feh_cannon', path='tree_feh_branches_leaves.png', leaves_only=True)
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'teff_cannon', path='tree_teff_branches.png')
    # for abund in abund_cols_use:
    #     colorize_tree(tree_struct, galah_cannon_subset, abund, path='tree_abund_'+abund+'.png')
    #     colorize_tree_branches(tree_struct, galah_cannon_subset, abund, path='tree_abund_'+abund+'_branches.png')


# ------------------------------------------------------------------
# -------------------- Tree analysis begin -------------------------
# ------------------------------------------------------------------
if not perform_data_analysis:
    raise SystemExit

# determine distances between pairs of repeated observations of the same object
if investigate_repeated:
    print 'Repeated obs graph distances'
    snr_cols = list(['snr_c1_guess', 'snr_c2_guess', 'snr_c3_guess', 'snr_c4_guess'])
    topology_dist = list([])
    # get repeated observations still found in the dataset
    id_uniq, id_count = np.unique(galah_cannon_subset['galah_id'], return_counts=True)
    id_uniq = id_uniq[np.logical_and(id_uniq > 0, id_count >= 2)]
    n_reps = len(id_uniq)
    print ' Number of repeats found: '+str(n_reps)
    if n_reps > 0:
        # output results
        if save_results:
            txt_file = open('repeted_obs_dist.txt', 'w')
        for galah_id in id_uniq:
            s_ids_repeated = galah_cannon_subset['sobject_id'][galah_cannon_subset['galah_id'] == galah_id]._data
            out_str = str(galah_id)+':'
            for s_ids_comp in itertools.combinations(s_ids_repeated, 2):
                if s_ids_comp[0] == s_ids_comp[1]:
                    dist = 0
                else:
                    dist = tree_struct.get_distance(str(s_ids_comp[0]), str(s_ids_comp[1]), topology_only=True)
                    snr_0 = galah_cannon_subset[galah_cannon_subset['sobject_id'] == s_ids_comp[0]][snr_cols].to_pandas().values[0]
                    snr_1 = galah_cannon_subset[galah_cannon_subset['sobject_id'] == s_ids_comp[1]][snr_cols].to_pandas().values[0]
                    snr_diff = np.abs(snr_0 - snr_1)
                    out_str += ' '+str(int(dist))+' (snr dif:'+str(snr_diff)+')'
                topology_dist.append(dist)
            if save_results:
                txt_file.write(out_str+'\n')
            else:
                print out_str
        # mean topology distance of all repeated observations - aka quality of the tree
        mean_topology_dist = np.mean(topology_dist)
        n_neighbours = np.sum(np.array(topology_dist) == 1)
        if len(topology_dist) > 0:
            if save_results:
                txt_file.write('Mean distance: ' + str(mean_topology_dist) + '\n')
                txt_file.write('Neighbours:   ' + str(n_neighbours) + '  ' + str(100.*n_neighbours/len(topology_dist)) + '%\n')
                n_neighbours_2 = np.sum(np.array(topology_dist) <= 2)
                n_neighbours_3 = np.sum(np.array(topology_dist) <= 3)
                n_neighbours_4 = np.sum(np.array(topology_dist) <= 4)
                n_neighbours_5 = np.sum(np.array(topology_dist) <= 5)
                txt_file.write('Neighbours 2: ' + str(n_neighbours_2) + '  ' + str(100. * n_neighbours_2 / len(topology_dist)) + '%\n')
                txt_file.write('Neighbours 3: ' + str(n_neighbours_3) + '  ' + str(100. * n_neighbours_3 / len(topology_dist)) + '%\n')
                txt_file.write('Neighbours 4: ' + str(n_neighbours_4) + '  ' + str(100. * n_neighbours_4 / len(topology_dist)) + '%\n')
                txt_file.write('Neighbours 5: ' + str(n_neighbours_5) + '  ' + str(100. * n_neighbours_5 / len(topology_dist)) + '%\n')

                txt_file.close()
            else:
                print mean_topology_dist
                print n_neighbours, 100.*n_neighbours/len(topology_dist)
        # temporally added exit as further analysis is not needed/wanted
        raise SystemExit
    else:
        print 'NOTE: No repeated observations found in the dataset'

print 'Traversing tree leaves - find mayor branches splits'
nodes_to_investigate = list([])
for t_node in tree_struct.traverse():
    if t_node.name == '':
        if is_node_before_leaves(t_node, min_leaves=2):
            # find ouh how far into the tree you can go before any mayor tree split happens
            n_objects_up = 2
            max_add_objects = 6
            ancestor_nodes = t_node.get_ancestors()
            for i_a in range(len(ancestor_nodes)):
                ancestor_obj_names = get_decendat_sobjects(ancestor_nodes[i_a])
                n_ancestor_obj_names = len(ancestor_obj_names)
                if n_ancestor_obj_names >= n_objects_up + max_add_objects:
                    if n_objects_up < 2:
                        # skip investigation of clusters with only 2 members
                        break
                    # print n_objects_up
                    if i_a > 0:
                        nodes_to_investigate.append(ancestor_nodes[i_a-1])
                    else:
                        nodes_to_investigate.append(t_node)
                    break
                else:
                    n_objects_up = n_ancestor_obj_names

# determine unique nodes to be investigated
# nodes_to_investigate = np.unique(nodes_to_investigate)

# much more consistent way of producing repeatable list of nodes to be visited for analysis
print 'Removing repeated nodes from the list (starting number of nodes is '+str(len(nodes_to_investigate))+')'
nodes_to_remove = list([])
for node_cur in nodes_to_investigate:
    idx_nodes = np.where(np.in1d(nodes_to_investigate, node_cur))[0]
    n_rep = len(idx_nodes)
    if n_rep > 1:
        nodes_to_remove.append(idx_nodes[1:])
nodes_to_investigate = np.delete(nodes_to_investigate, np.unique(np.hstack(nodes_to_remove)))

# colorize tree structures/nodes/branches/leaves/something that were/will be evaluated, test for selection criteria
print 'Analyzing selected nodes and leaves, plotting graph for selection'
sobjects_analyzed_all = list([])
# gather all unique sobject_ids
for i_node in range(len(nodes_to_investigate)):
    descendants = get_decendat_sobjects(nodes_to_investigate[i_node])
    for cur_des in descendants:
        sobjects_analyzed_all.append(cur_des)
sobjects_analyzed_all = np.unique(np.array(sobjects_analyzed_all))
# plot graph
mark_objects(tree_struct, np.int64(sobjects_analyzed_all), path='analysis_selected_leaves.png')

print 'Final number of nodes to be investigated is: ', len(nodes_to_investigate)
for i_node in range(len(nodes_to_investigate)):
    descendants = get_decendat_sobjects(nodes_to_investigate[i_node])
    start_gui_explorer(descendants, manual=manual_GUI_investigation, initial_only=False, loose=loose_pairs,
                       save_dir=trees_dir+final_dir, i_seq=i_node, kinematics_source=tgas_ucac5_use)

z