import imp, sys
import itertools

import numpy as np
import astropy.units as un
import astropy.coordinates as coord
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, scale
from scipy.cluster.hierarchy import linkage, to_tree
from astropy.table import Table, join
from getopt import getopt

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import *
imp.load_source('helper2', '../tSNE_test/cannon3_functions.py')
from helper2 import *
imp.load_source('norm', '../Stellar_parameters_interpolator/data_normalization.py')
from norm import *

imp.load_source('distf', '../tSNE_test/distances.py')
from distf import *
imp.load_source('tsne_functions', '../tSNE_test/tsne_functions.py')
from tsne_functions import *

from colorize_tree import *
from filter_galah import *
from NJ_tree_analysis_functions import *

from ete3 import Tree
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path


# ------------------------------------------------------------------
# -------------------- Functions -----------------------------------
# ------------------------------------------------------------------
# relocated for general use to the NJ_tree_analysis_functions.py

# ------------------------------------------------------------------
# -------------------- Program settings ----------------------------
# ------------------------------------------------------------------
sys.setrecursionlimit(5000)  # correction for deeper trees (RuntimeError: maximum recursion depth exceeded while calling a Python object)
# input arguments
input_arguments = sys.argv

# data settings
join_repeated_obs = False
normalize_abund = True
weights_abund = True
plot_overall_graphs = True
perform_data_analysis = True
investigate_repeated = True
save_results = True
manual_GUI_investigation = False
tgas_ucac5_use = 'ucac5'  # valid options are 'tgas', 'ucac5' and 'gaia'(when available)

# positional filtering settings
linkage_metric = 'cityblock'
linkage_method = 'weighted'

loose_pairs = True
filter_by_sky_position = True
tsne_run = False
# command line parameters parser
if len(input_arguments) > 3:
    print 'Using arguments given from command line'
    ra_center = float(input_arguments[1])
    dec_center = float(input_arguments[2])
    position_radius = float(input_arguments[3])
    # select only input arguments including -- string
    input_options = [arg for arg in input_arguments if '--' in arg]
    if len(input_options) > 0:
        opts, args = getopt(input_options, '', ['metric=', 'join=', 'loose=', 'method=', 'tsne='])
        # set parameters, depending on user inputs
        print input_arguments
        print opts
        for o, a in opts:
            if o == '--metric':
                linkage_metric = a
            if o == '--method':
                linkage_method = a
            if o == '--join':
                join_repeated_obs = string_bool(a)
            if o == '--loose':
                loose_pairs = string_bool(a)
            if o == '--tsne':
                tsne_run = string_bool(a)

else:
    ra_center = 30.  # degrees
    dec_center = 0.  # degrees
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

galah_cannon = Table.read(galah_data_dir+'sobject_iraf_iDR2_171103_cannon.fits')
abund_cols = get_abundance_cols3(galah_cannon.colnames)
abund_cols = remove_abundances(abund_cols, what_abund_remove(), type='cannon')

# galah_cannon = Table.read(galah_data_dir+'galah_abund_ANN_SME3.0.1_stacked_median.fits')
# abund_cols = get_abundance_colsann(galah_cannon.colnames)
# abund_cols = remove_abundances(abund_cols, what_abund_remove(), type='ann')

# additional kinematics data
if tgas_ucac5_use is 'tgas':
    galah_kinematics_xmatch = Table.read(galah_data_dir + 'galah_tgas_xmatch.csv')['sobject_id', 'pmra', 'pmdec', 'parallax']
elif tgas_ucac5_use is 'ucac5':
    galah_kinematics_xmatch = Table.read(galah_data_dir + 'galah_cannon_3.0_ucac5_joined_20171111.fits')['sobject_id', 'pmra', 'pmdec']

# known stars in clusters in this dataset
# stars_cluster_data = Table.read(galah_data_dir+'clusters/galah_cluster_members_merged_galahonly_20171111.fits')
stars_cluster_data = Table.read(galah_data_dir+'clusters/2m_all_clusters_20171111_xmatch_probable.fits')
# clusters_ra, clusters_dec = define_cluster_centers(stars_cluster_data, galah_cannon)

# get cannon abundance cols
abund_elem = get_element_names(abund_cols)
abund_cols_use = list(abund_cols)
abund_cols_f = ['flag_'+col for col in abund_cols]
abund_cols_f_use = list(abund_cols_f)

# create a filter that will be used to create a subset of data used in  abundance analysis
print 'Input data lines: '+str(len(galah_cannon))
print 'Creating filters for data subset'
galah_filter_ok = FILTER(verbose=True)
# remove data from the initial observations
print 'Commissioning and test nights'
galah_filter_ok.filter_attribute(galah_cannon, attribute='sobject_id', value=140301000000000, comparator='>')
# UPDATED DATES: remove commissioning spectra and first few test nights

# #########
# # NOTE: temporary test for clustering of data with the same
# # general flags statistics
# flags_data = galah_cannon[abund_cols_f].to_pandas().values
# flags_data = np.int64(flags_data <= 3)
# flags_weights = 2**np.arange(len(abund_cols_f))  # kind of binary flag set description
# flags_data *= flags_weights
# flags_data_sum = np.sum(flags_data, axis=1)
# u_flag_w, u_flag_w_counts = np.unique(flags_data_sum, return_counts=True)
# # abund and object filtering based on those inforation
# idx_u_flag_w = np.argsort(u_flag_w_counts)[-4]
# idx_objects = np.logical_and(flags_data_sum == u_flag_w[idx_u_flag_w],
#                              np.isfinite(galah_cannon[abund_cols_use].to_pandas().values).all(axis=1))
# idx_cols = np.where(flags_data[np.where(idx_objects)[0][0], :] > 0)
# flags_data = None
# # do the actuall subseting of datasets
# abund_cols_use = list(np.array(abund_cols_use)[idx_cols])
# abund_cols_f_use = list(np.array(abund_cols_f_use)[idx_cols])
# galah_filter_ok._merge_ok(idx_objects)
# print abund_cols_use, abund_cols_f
# # NOTE: test end
# #########

# ------------------------------------------------------------------
# -------------------- Abundance data filtering --------------------
# ------------------------------------------------------------------
# first remove all somehow flagged observation
print 'Red , cannon and guess flags'
idx_ok_flags = quality_flagging(galah_cannon, cannon_flag=False, chi2outliers=True, return_idx=True)
galah_filter_ok._merge_ok(idx_ok_flags)
# galah_filter_ok._merge_ok(galah_cannon['red_flag'] == 0)
# galah_filter_ok._merge_ok(galah_cannon['flag_guess'] == 0)

# filter by cannon CHI2 value
# about 35-40k for 1.2 and 75k for 2.1.7
# galah_filter_ok.filter_attribute(galah_cannon, attribute='chi2_cannon', value=35000, comparator='<')
print 'SNR C2 cut'
galah_filter_ok.filter_attribute(galah_cannon, attribute='snr_c2_iraf', value=20, comparator='>')

# filter out classes of possibly problematic stars as detected by Gregor
# determine which problematic stars might be problematic for abundance determination
# NOTE: abundance data is already flagged by Sven for this kind of problematic objects

# filter out data that are not in kinematics dataset
print 'Kinematics cut'
suffix += '_'+tgas_ucac5_use
galah_filter_ok.filter_objects(galah_cannon, galah_kinematics_xmatch, identifier='sobject_id', invert=False)

# select only object with enough unglagged abundances
print 'Only enough unflagged abundances'
galah_filter_ok._merge_ok(object_with_min_abundances(galah_cannon, abund_cols_f_use, len(abund_cols_f_use)/2))

# create a subset defined by filters above
galah_cannon_subset = galah_filter_ok.apply_filter(galah_cannon)

# ------------------------------------------------------------------
# -------------------- Join repeated observations ------------------
# ------------------------------------------------------------------
if join_repeated_obs:
    # TODO: export repeated observations
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
    print abund_cols_use
    cannon_flags = galah_cannon_subset[abund_cols_f_use].to_pandas().values
    cannon_flags = cannon_bad_flags(cannon_flags)
    # galah_cannon_subset_abund[cannon_flags] = np.nan
    cannon_data_means = np.nanmean(galah_cannon_subset_abund, axis=0)
    cannon_data_stds = np.nanstd(galah_cannon_subset_abund, axis=0)
    galah_cannon_subset_abund = (galah_cannon_subset_abund - cannon_data_means) / cannon_data_stds
    # print cannon_data_means, cannon_data_stds
    cannon_data_means[:] = 0

# apply weights to the normalized abund parameters if requested to do so
if weights_abund:
    suffix += '_weight'
    print 'Abundance weighting'
    for i_col in range(len(abund_cols_use)):
        ab_multi = abund_multiplier_c3(abund_elem[i_col])
        print ' ', abund_elem[i_col], ab_multi
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
    suffix += '_'+linkage_method
    # distance computation
    suffix += '_'+linkage_metric
if loose_pairs:
    suffix += '_loose'

final_dir = 'NJ_clusters'
final_dir += '_cannon_3.0'
# final_dir += '_ANN'

final_dir += '_abundflags'+suffix+suffix_pos
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

# create a map of used objects and mark the ones that might be cluster members
print 'Plotting objects on sky'
idx_in = np.in1d(galah_cannon_subset['sobject_id'], stars_cluster_data['sobject_id'])
plt.scatter(galah_cannon_subset['ra'][idx_in], galah_cannon_subset['dec'][idx_in], lw=0, s=2.5, c='red')
plt.scatter(galah_cannon_subset['ra'], galah_cannon_subset['dec'], lw=0, s=1, c='black')
plt.savefig('area_sky_objects.png', dpi=500)
plt.close()

if tsne_run:
    tsne_csv_out = 'tsne.fits'
    if os.path.isfile(tsne_csv_out):
        tsne_out = Table.read(tsne_csv_out)
    else:
        print 'Computing tsne'
        tsne_class = TSNE(n_components=2, perplexity=40.0, n_iter=1000, n_iter_without_progress=350, init='random', verbose=1,
                          method='barnes_hut', angle=0.3, metric='precomputed', min_grad_norm=1e-07, learning_rate=150.0)
        dist_values = cityblock_nans(galah_cannon_subset_abund, triu=False)
        print dist_values [:5,:5]
        tsne_res = tsne_class.fit_transform(dist_values)

        # output tsne plot(s)
        plt.scatter(tsne_res[:, 0], tsne_res[:, 1], lw=0, c='black', s=1)
        plt.savefig('tsne.png', dpi=500)
        plt.close()
        for cluster in np.unique(stars_cluster_data['cluster']):
            cluster_targets = stars_cluster_data[stars_cluster_data['cluster'] == cluster]['sobject_id']
            idx_cluster = np.in1d(galah_cannon_subset['sobject_id'], cluster_targets)
            if np.sum(idx_cluster) <= 0:
                continue
            plt.scatter(tsne_res[:, 0][idx_cluster], tsne_res[:, 1][idx_cluster], lw=0, c='red', s=5)
            plt.scatter(tsne_res[:, 0], tsne_res[:, 1], lw=0, c='black', s=1)
            plt.savefig('tsne_'+cluster+'.png', dpi=500)
            plt.close()

        # output data
        tsne_out = galah_cannon_subset['sobject_id', 'galah_id']
        tsne_out['tsne_axis_1'] = tsne_res[:, 0]
        tsne_out['tsne_axis_2'] = tsne_res[:, 1]
        tsne_out.write(tsne_csv_out)

    # manual investigation of computed tSNE projection
    # fig, ax = plt.subplots(1, 1)
    # pts = ax.scatter(tsne_out['tsne_axis_1'], tsne_out['tsne_axis_2'], lw=0, c='black', s=4)
    # selector = PointSelector(ax, tsne_out)
    # plt.show()
    # plt.close()


# ------------------------------------------------------------------
# -------------------- Tree computation ----------------------------
# ------------------------------------------------------------------

# compute distances and create phylogenetic tree from distance information
if not os.path.isfile(output_nwm_file):
    print 'Computing data distances'
    if hierachical_scipy:
        print 'Hierarchical clustering started'
        if linkage_metric == 'esd':
            linkage_matrix = linkage(esd_dist_compute(galah_cannon_subset_abund, means=cannon_data_means, stds=cannon_data_stds, triu=True), method=linkage_method)
        elif linkage_metric == 'cityblocknans':
            linkage_matrix = linkage(cityblock_nans(galah_cannon_subset_abund, triu=True), method=linkage_method)
        elif linkage_metric == 'cosinedistnans':
            linkage_matrix = linkage(cosinedist_nans(galah_cannon_subset_abund, triu=True), method=linkage_method)
        else:
            linkage_matrix = linkage(galah_cannon_subset_abund, method=linkage_method, metric=linkage_metric)  # might use too much RAM for large arrays
        linkage_tree = to_tree(linkage_matrix, False)
        newic_tree_str = getNewick(linkage_tree, "", linkage_tree.dist, galah_cannon_subset['sobject_id'].data)
        nwm_txt = open(output_nwm_file, 'w')
        nwm_txt.write(newic_tree_str)
        nwm_txt.close()

# read output tree file
tree_struct = get_tree_from_file(output_nwm_file)

if plot_overall_graphs:
    print 'Plotting graphs'
    for cluster in np.unique(stars_cluster_data['cluster']):
        print cluster
        cluster_targets = stars_cluster_data[stars_cluster_data['cluster'] == cluster]['sobject_id']
        mark_objects(tree_struct, cluster_targets, path='cluster_'+cluster+'.png')
    # colorize_tree(tree_struct, galah_cannon_subset, 'Logg_cannon', path='tree_logg.png')
    # colorize_tree(tree_struct, galah_cannon_subset, 'Feh_cannon', path='tree_feh.png')
    # colorize_tree(tree_struct, galah_cannon_subset, 'Teff_cannon', path='tree_teff.png')
    # colorize_tree_branches(tree_struct, galah_cannon_subset, 'Logg_cannon', path='tree_logg_branches.png')
    # colorize_tree_branches(tree_struct, galah_cannon_subset, 'Feh_cannon', path='tree_feh_branches.png')
    # colorize_tree_branches(tree_struct, galah_cannon_subset, 'Feh_cannon', path='tree_feh_branches_leaves.png', leaves_only=True)
    # colorize_tree_branches(tree_struct, galah_cannon_subset, 'Teff_cannon', path='tree_teff_branches.png')
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
    snr_cols = list(['snr_c1_iraf', 'snr_c2_iraf', 'snr_c3_iraf', 'snr_c4_iraf'])
    topology_dist = list([])
    # get repeated observations still found in the dataset
    id_uniq, id_count = np.unique(galah_cannon_subset['galah_id'], return_counts=True)
    id_uniq = id_uniq[np.logical_and(id_uniq > 0, id_count >= 2)]
    id_uniq = id_uniq[id_uniq != 999999]  # something strange happens for this value
    n_reps = len(id_uniq)
    print ' Number of repeats found: '+str(n_reps)
    if n_reps > 0:
        # output results
        if save_results:
            txt_file = open('repeted_obs_dist.txt', 'w')
        for galah_id in id_uniq:
            s_ids_repeated = galah_cannon_subset['sobject_id'][galah_cannon_subset['galah_id'] == galah_id]
            out_str = str(galah_id)+':'
            for s_ids_comp in itertools.combinations(list(s_ids_repeated), 2):
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
        # raise SystemExit
    else:
        print 'NOTE: No repeated observations found in the dataset'

print 'Traversing tree leaves - find mayor branches splits'
nodes_to_investigate = list([])
for t_node in tree_struct.traverse():
    if t_node.name == '':
        if is_node_before_leaves(t_node, min_leaves=2):
            # find ouh how far into the tree you can go before any mayor tree split happens
            split_node = get_major_split(t_node)
            nodes_to_investigate.append(split_node)

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

# check if node is already inbeded in higher one
print 'Removing imbeded tree cuts'
tree_root = tree_struct.get_tree_root()
nodes_to_remove = list([])
n_nodes = len(nodes_to_investigate)
for i_n in range(0, n_nodes):
    for j_n in range(i_n+1, n_nodes):
        node_cur1 = nodes_to_investigate[i_n]
        node_cur2 = nodes_to_investigate[j_n]
        com_anc = node_cur1.get_common_ancestor(node_cur2)
        if com_anc == node_cur1 or com_anc == node_cur2:
            # delete one that is father from the root
            r_dist1 = tree_root.get_distance(node_cur1)
            r_dist2 = tree_root.get_distance(node_cur2)
            if r_dist1 < r_dist2:
                nodes_to_remove.append(j_n)
            else:
                nodes_to_remove.append(i_n)
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

