import os, imp, time
import itertools

import numpy as np
import astropy.units as un
import astropy.coordinates as coord

from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.preprocessing import StandardScaler, scale
from scipy.cluster.hierarchy import linkage, to_tree
from skbio import DistanceMatrix
from skbio.tree import nj
from astropy.table import Table, join

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir, get_abundance_cols, define_cluster_centers, determine_clusters_abundance_deviation, get_multipler
imp.load_source('norm', '../Stellar_parameters_interpolator/data_normalization.py')
from norm import *

imp.load_source('distf', '../tSNE_test/distances.py')
from distf import *

from colorize_tree import *
from filter_galah import *
from NJ_tree_analysis_functions import *

from ete3 import Tree


# ------------------------------------------------------------------
# -------------------- Functions -----------------------------------
# ------------------------------------------------------------------
def start_gui_explorer(objs, manual=True, save_dir='', i_seq=1, kinematics_source=''):
    code_path = '/home/klemen/tSNE_test/'
    # temp local check to set the correct directory when not run from gigli pc
    if not os.path.exists(code_path):
        code_path = '/home/klemen/gigli_mount/tSNE_test/'
    if manual:
        manual_suffx='_manual'
    else:
        manual_suffx = '_auto'
    # crete filename that is as unique as possible
    out_file = code_path + 'tree_temp_'+str(i_seq)+manual_suffx+'_'+str(int(time.time()))+'.txt'
    txt = open(out_file, 'w')
    txt.write(','.join(objs))
    txt.close()
    if manual:
        exec_str = '/home/klemen/anaconda2/bin/python '+code_path+'GUI_abundance_kinematics_analysis.py ' + out_file
    else:
        exec_str = '/home/klemen/anaconda2/bin/python '+code_path+'GUI_abundance_kinematics_analysis_automatic.py '
        exec_str += out_file + ' ' + save_dir+'/node_{:03d}'.format(i_seq)
    # add kinematics use information
    exec_str += ' '+kinematics_source
    # execute GUI explorer or automatic analysis
    os.system(exec_str)
    # remove file with sobject_ids
    os.remove(out_file)


# https://stackoverflow.com/questions/28222179/save-dendrogram-to-newick-format
def getNewick(node, newick, parentdist, leaf_names):
    if node.is_leaf():
        return "%s:%.8f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.8f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = getNewick(node.get_left(), newick, node.dist, leaf_names)
        newick = getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick

# ------------------------------------------------------------------
# -------------------- Program settings ----------------------------
# ------------------------------------------------------------------
# data settings
join_repeated_obs = False
normalize_abund = True
weights_abund = False
plot_overall_graphs = True
perform_data_analysis = True
investigate_repeated = True
save_results = True
manual_GUI_investigation = False
tgas_ucac5_use = 'ucac5'  # valid options are 'tgas', 'ucac5' and 'gaia'(when available)
# positional filtering settings
filter_by_sky_position = True
ra_center = 57.  # degrees
dec_center = 24.  # degrees
position_radius = 25.  # degrees
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
# galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon2.1.7.fits')
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
stars_cluster_data = Table.read(galah_data_dir+'sobject_clusterstars_1.0.fits')
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

# use for Cannon 1.2
# filter by cannon set flags - parameters flag, must be set to 0 for valid data
galah_filter_ok._merge_ok(galah_cannon['flag_cannon'] == '0.0')

# use for Cannon 2.1.7
# filter by cannon set flags - abundances flags, all flags must be set to 0 for valid data
# for abund_col in abund_cols_use:
#     galah_filter_ok._merge_ok(galah_cannon['flag_'+abund_col] == 0)

# remove rows with any nan values in any of the abundance columns/attributes
galah_filter_ok.filter_valid_rows(galah_cannon, cols=abund_cols_use)

# filter by CCD SNR values
# galah_filter_ok.filter_attribute(galah_cannon, attribute='snr2_c1_iraf', value=20, comparator='>')

# filter by cannon CHI2 value
# about 35-40k for 1.2 and 75k for 2.1.7
galah_filter_ok.filter_attribute(galah_cannon, attribute='chi2_cannon', value=35000, comparator='<')
galah_filter_ok.filter_attribute(galah_cannon, attribute='snr_c1_iraf', value=15, comparator='>')

# filter out problematic stars as detected by Gregor
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
    std_abundances = determine_clusters_abundance_deviation(_temp_sub, stars_cluster_data, abund_cols)
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
    compute_mean_cols = list(abund_cols)
    compute_mean_cols.append('rv_guess')
    i_rep, c_rep = np.unique(galah_cannon_subset['galah_id'], return_counts=True)
    ids_join = i_rep[np.logical_and(i_rep > 0, c_rep >= 2)]
    n_reps = np.sum(c_rep[np.logical_and(i_rep > 0, c_rep >= 2)])
    if len(ids_join) > 0:
        print 'Number of repeated object: '+str(len(ids_join))+' and observations: '+str(n_reps)
        for id_join in ids_join:
            idx_rows = np.where(galah_cannon_subset['galah_id'] == id_join)
            mean_cols = np.median(galah_cannon_subset[idx_rows][compute_mean_cols].to_pandas().values, axis=0)
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
    for i_col in range(len(abund_cols)):
        ab_multi = get_multipler(std_abundances[i_col])
        print ' ', abund_cols[i_col], ab_multi, std_abundances[i_col]
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
    suffix += '_manhattan'

final_dir = 'NJ_tree_cannon_1.2_mainrun_abundflags_chi2_prob'+suffix+suffix_pos
# final_dir = 'NJ_tree_cannon_1.2_mainrun_abundflags_chi2_prob_tgas_norep_norm_megacc_manhattan'
move_to_dir(final_dir)

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
        linkage_matrix = linkage(galah_cannon_subset_abund, method='weighted', metric='cityblock')  # might use too much RAM
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
    for cluster in np.unique(stars_cluster_data['cluster_name']):
        print cluster
        cluster_targets = stars_cluster_data[stars_cluster_data['cluster_name'] == cluster]['sobject_id']
        mark_objects(tree_struct, cluster_targets, path='cluster_'+cluster+'.png')
    # colorize_tree(tree_struct, galah_cannon_subset, 'logg_cannon', path='tree_logg.png')
    # colorize_tree(tree_struct, galah_cannon_subset, 'feh_cannon', path='tree_feh.png')
    # colorize_tree(tree_struct, galah_cannon_subset, 'teff_cannon', path='tree_teff.png')
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'logg_cannon', path='tree_logg_branches.png')
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'feh_cannon', path='tree_feh_branches.png')
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'feh_cannon', path='tree_feh_branches_leaves.png', leaves_only=True)
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'teff_cannon', path='tree_teff_branches.png')
    # for abund in abund_cols:
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
    if save_results:
        txt_file = open('repeted_obs_dist.txt', 'w')
    id_uniq, id_count = np.unique(galah_cannon_subset['galah_id'], return_counts=True)
    for galah_id in id_uniq[id_count >= 2]:
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
            txt_file.write('Neighbours: ' + str(n_neighbours) + '  ' + str(100.*n_neighbours/len(topology_dist)) + '%\n')
            txt_file.close()
        else:
            print mean_topology_dist
            print n_neighbours, 100.*n_neighbours/len(topology_dist)
    else:
        print 'NOTE: No repeated observations found in the dataset'

print 'Traversing tree leaves - find mayor branches splits'
nodes_to_investigate = list([])
for t_node in tree_struct.traverse():
    if t_node.name == '':
        if is_node_before_leaves(t_node, min_leaves=2):
            # find ouh how far into the tree you can go before any mayor tree split happens
            n_objects_up = 2
            max_add_objects = 5
            ancestor_nodes = t_node.get_ancestors()
            for i_a in range(len(ancestor_nodes)):
                ancestor_obj_names = get_decendat_sobjects(ancestor_nodes[i_a])
                n_ancestor_obj_names = len(ancestor_obj_names)
                if n_ancestor_obj_names >= n_objects_up + max_add_objects:
                    if n_objects_up < 2:
                        # skip investigation of clusters with only 2 members
                        break
                    print n_objects_up
                    if i_a > 0:
                        nodes_to_investigate.append(ancestor_nodes[i_a-1])
                    else:
                        nodes_to_investigate.append(t_node)
                    break
                else:
                    n_objects_up = n_ancestor_obj_names
# determine unique nodes to be investigated
nodes_to_investigate = np.unique(nodes_to_investigate)
print 'Final number of nodes to be investigated is: ', len(nodes_to_investigate)
for i_node in range(len(nodes_to_investigate)):
    descendants = get_decendat_sobjects(nodes_to_investigate[i_node])
    start_gui_explorer(descendants, manual=manual_GUI_investigation,
                       save_dir=trees_dir+final_dir, i_seq=i_node, kinematics_source=tgas_ucac5_use)


raise SystemExit
print 'Traversing tree leaves - min number of objects in leaves cluster'
i_t = 0
for t_node in tree_struct.traverse():
    i_t += 1
    if i_t < 1500:
        continue
    if t_node.name == '':
        descendants = get_decendat_sobjects(t_node)
        n_descendants = len(descendants)
        if n_descendants < 25 and n_descendants > 20:
            start_gui_explorer(descendants, manual=manual_GUI_investigation,
                               save_dir=trees_dir + final_dir, i_seq=i_node, kinematics_source=tgas_ucac5_use)


raise SystemExit
# start traversing the tree
print 'Traversing tree leaves - tree final leaves only'

# join datasets
actions_data = Table.read(galah_data_dir+'galah_tgas_xmatch_actions.fits')['sobject_id', 'J_R', 'L_Z', 'J_Z', 'Omega_R', 'Omega_Phi', 'Omega_Z', 'Theta_R', 'Theta_Phi', 'Theta_Z',
                                                                                     'ra_gaia', 'dec_gaia', 'parallax', 'pmra', 'pmdec']
# actions_data = actions_data[np.isfinite(actions_data['J_R'])].filled()
galah_tgas = join(galah_cannon_subset, actions_data, keys='sobject_id', join_type='inner')

if save_results:
    txt_file = open('leaves_obs.txt', 'w')
for t_node in tree_struct.traverse():
    if t_node.name == '':
        if is_node_before_leaves(t_node, min_leaves=2):
            s_obj_names = get_decendat_sobjects(t_node)
            galah_tgas_sub = get_data_subset(galah_tgas, s_obj_names, select_by='sobject_id')
            xyz_gal_pos, rpz_gal_pos, uvw_gal_vel = predict_stream_description(galah_tgas_sub, xyz_out=True)
            if evaluate_pairwise_distances(uvw_gal_vel, median=True, meassure='manh') < 20:
                if galah_tgas_sub[0]['galah_id'] == galah_tgas_sub[1]['galah_id']:
                    out_text_line = '--------    Repeated obs:    ---------\nsobject_ids: ' + ' '.join(s_obj_names) + '\n'
                    if save_results:
                        txt_file.write(out_text_line)
                    else:
                        print out_text_line
                    continue
                n_in_clusters = objects_in_list(np.int64(s_obj_names), stars_cluster_data['sobject_id'])
                if n_in_clusters > 0:
                    out_text_line = '-----    Known ('+str(n_in_clusters)+') cluster obj  ------'
                    if save_results:
                        txt_file.write(out_text_line+'\n')
                    else:
                        print out_text_line
                if save_results:
                    txt_file.write('sobject_ids: '+' '.join(s_obj_names) + '\n')
                    txt_file.write('XYZ pos:\n' + str(xyz_gal_pos) + '\n')
                    txt_file.write('RPZ pos:\n' + str(rpz_gal_pos) + '\n')
                    txt_file.write('UVW vel:\n' + str(uvw_gal_vel) + '\n')
                    txt_file.write('\n')
                else:
                    print s_obj_names
                    print xyz_gal_pos
                    print rpz_gal_pos
                    print uvw_gal_vel
                    print 'Space dist ', evaluate_pairwise_distances(xyz_gal_pos, median=True, meassure='eucl')
                    print 'Veloc dist ', evaluate_pairwise_distances(uvw_gal_vel, median=True, meassure='manh')
                    print
                # repeat the same analysis from ancestors point of view
                for ancestor_node in t_node.get_ancestors()[0:1]:  # for now investigate only from the first ancestor
                    s_obj_names = get_decendat_sobjects(ancestor_node)
                    galah_tgas_sub = get_data_subset(galah_tgas, s_obj_names, select_by='sobject_id')
                    xyz_gal_pos, rpz_gal_pos, uvw_gal_vel = predict_stream_description(galah_tgas_sub, xyz_out=True)
                    # if evaluate_angles(star_angles) > 75 and evaluate_pairwise_distances(star_xyz_vel, median=True) < 20:
                    if save_results:
                        txt_file.write('sobject_ids: ' + ' '.join(s_obj_names) + '\n')
                        txt_file.write('XYZ pos:\n' + str(xyz_gal_pos) + '\n')
                        txt_file.write('RPZ pos:\n' + str(rpz_gal_pos) + '\n')
                        txt_file.write('UVW vel:\n' + str(uvw_gal_vel) + '\n')
                        txt_file.write('\n')
                    else:
                        print '-- >Second step, from ancestors point of view'
                        print s_obj_names
                        print xyz_gal_pos
                        print rpz_gal_pos
                        print uvw_gal_vel
                        print 'Space dist ', evaluate_pairwise_distances(xyz_gal_pos, median=True, meassure='eucl')
                        print 'Veloc dist ', evaluate_pairwise_distances(uvw_gal_vel, median=True, meassure='manh')
                        print

                    # if len(node_desc) == 2:  # get final branches
        #     s_obj = list([])
        #     for node in node_desc:
        #         node_name = node.name
        #         if node_name != '':
        #             s_obj.append(node_name)
        #     if not (np.array(s_obj) == '').any():
        #         idx_galah_tgas = np.where(np.in1d(galah_tgas['sobject_id'], np.int64(s_obj)))
        #         # node_data = galah_tgas[idx_galah_tgas]['J_R', 'L_Z', 'J_Z'].to_pandas().values
        #         if len(idx_galah_tgas[0]) == 2:
        #             # node_dist = np.sum(np.abs(node_data[0]-node_data[1]))
        #             # compute xyz velocities
        #             galah_tgas_sub = galah_tgas[idx_galah_tgas]
        #             xyz_pos = coord.SkyCoord(ra=galah_tgas_sub['ra_gaia']*un.deg,
        #                                      dec=galah_tgas_sub['dec_gaia']*un.deg,
        #                                      distance=1e3/galah_tgas_sub['parallax']*un.pc).cartesian
        #             xyz_vel = motion_to_cartesic(galah_tgas_sub['ra_gaia'], galah_tgas_sub['dec_gaia'],
        #                                          galah_tgas_sub['pmra'],
        #                                          galah_tgas_sub['pmdec'], galah_tgas_sub['rv_guess'],
        #                                          plx=galah_tgas_sub['parallax']).T
        #             stream_pred = mean_velocity(xyz_vel)
        #             stream_plane_intersect = stream_plane_vector_intersect(xyz_pos, xyz_vel, stream_pred)
        #             intersect_dist = points_distance(stream_plane_intersect[0], stream_plane_intersect[1])
        #             vel_dist = np.sum(np.abs(xyz_vel[0] - xyz_vel[1]))
        #             # if node_dist < 0.02:
        #             #     print s_obj
        #             if intersect_dist < 500 and vel_dist < 20:
        #                 if galah_tgas_sub[0]['galah_id'] == galah_tgas_sub[1]['galah_id']:
        #                     repeated_text = '  - repeated observation of object '+str(galah_tgas_sub[0]['galah_id'])
        #                 else:
        #                     repeated_text = ''
        #                 if save_results:
        #                     txt_file.write('sobject_ids: '+' '.join(s_obj) + '\n'+repeated_text+'\n')
        #                     txt_file.write('Dist veloc: '+str(vel_dist) + '\n')
        #                     txt_file.write('Dist plane: '+str(intersect_dist) + '\n')
        #                     txt_file.write('XYZ velocity:\n' + str(xyz_vel) + '\n')
        #                     txt_file.write('Plane intersect:\n' + str(stream_plane_intersect) + '\n')
        #                     txt_file.write('\n')
        #                 else:
        #                     print s_obj,repeated_text
        #                     print 'Dist veloc: ', vel_dist
        #                     print 'Dist plane: ', intersect_dist
        #                     # print galah_tgas_sub['J_R', 'L_Z', 'J_Z', 'Omega_R', 'Omega_Phi', 'Omega_Z']
        #                     print 'XYZ velocity:\n', xyz_vel
        #                     print 'Plane intersect:\n', stream_plane_intersect
        #                     print ''
if save_results:
    txt_file.close()
