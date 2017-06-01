import os, imp, time

import numpy as np
import astropy.units as un
import astropy.coordinates as coord

from sklearn.metrics.pairwise import manhattan_distances
from skbio import DistanceMatrix
from skbio.tree import nj
from astropy.table import Table, join

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir, get_abundance_cols, define_cluster_centers, determine_clusters_abundance_deviation, get_multipler
imp.load_source('norm', '../Stellar_parameters_interpolator/data_normalization.py')
from norm import *

from colorize_tree import *
from filter_galah import *

from ete3 import Tree

# ------------------------------------------------------------------
# -------------------- Program settings ----------------------------
# ------------------------------------------------------------------

normalize_abund = True
weights_abund = False
plot_overall_graphs = False
perfrom_data_analysis = True
suffix = ''

# ------------------------------------------------------------------
# -------------------- Data reading and initial handling -----------
# ------------------------------------------------------------------

galah_data_dir = '/home/klemen/GALAH_data/'
nj_data_dir = '/home/klemen/NJ_tree_settings/'
# Galah parameters and abundance data
print 'Reading data'
# galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon2.1.7.fits')
galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')
galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')['sobject_id', 'rv_guess', 'teff_guess', 'logg_guess', 'feh_guess']
galah_tsne_class = Table.read(galah_data_dir+'tsne_class_1_0.csv')
galah_tgas_xmatch = Table.read(galah_data_dir+'galah_tgas_xmatch.csv')
galah_flats = Table.read(galah_data_dir+'flat_objects.csv')
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
# about 40k for 1.2 and 75k for 2.1.7
galah_filter_ok.filter_attribute(galah_cannon, attribute='chi2_cannon', value=40000, comparator='<')

# filter out problematic stars as detected by Gregor
galah_filter_ok.filter_objects(galah_cannon, galah_tsne_class, identifier='sobject_id')

# filter out flat target/object
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

# filter out data that are not in tgas set
suffix += '_tgas'
galah_filter_ok.filter_objects(galah_cannon, galah_tgas_xmatch, identifier='sobject_id', invert=False)

# ------------------------------------------------------------------
# -------------------- Final filtering and data preparation --------
# ------------------------------------------------------------------

# create a subset defined by filters above
galah_cannon_subset = galah_filter_ok.apply_filter(galah_cannon)
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

# for ra_center, dec_center in zip([],[]):
#
# convert ra and dec to astropy coordinates
# ra_center = 57.
# dec_center = 24.
use_megacc = True
phylip_shape = False

if phylip_shape:
    output_nwm_file = 'outtree'
    suffix += '_phylip'
elif use_megacc:
    output_nwm_file = 'distances_network.nwk'
    suffix += '_megacc'

move_to_dir('NJ_tree_cannon_1.2_mainrun_abundflags_chi2'+suffix)

# ------------------------------------------------------------------
# -------------------- Tree computation ----------------------------
# ------------------------------------------------------------------

# compute distances and create phylogenetic tree from distance information
if not os.path.isfile(output_nwm_file):
    print 'Computing data distances'
    distances = manhattan_distances(galah_cannon_subset_abund)

    # print distances[250:265, :15]
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
if os.path.isfile(output_nwm_file):
    txt = open(output_nwm_file, 'r')
    file_str = ''
    for line in txt:
        file_str += line[:-1]
    tree_struct = Tree(file_str)
    txt.close()
else:
    print 'ERROR: check megacc as it did not produce the following file '+output_nwm_file

if plot_overall_graphs:
    print 'Plotting graphs'
    for cluster in np.unique(stars_cluster_data['cluster_name']):
        print cluster
        cluster_targets = stars_cluster_data[stars_cluster_data['cluster_name'] == cluster]['sobject_id']
        mark_objects(tree_struct, cluster_targets, path='cluster_'+cluster+'.png')
    colorize_tree(tree_struct, galah_cannon_subset, 'logg_cannon', path='tree_logg.png')
    colorize_tree(tree_struct, galah_cannon_subset, 'feh_cannon', path='tree_feh.png')
    colorize_tree(tree_struct, galah_cannon_subset, 'teff_cannon', path='tree_teff.png')
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'logg_cannon', path='tree_logg_branches.png')
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'feh_cannon', path='tree_feh_branches.png')
    colorize_tree_branches(tree_struct, galah_cannon_subset, 'teff_cannon', path='tree_teff_branches.png')
    for abund in abund_cols:
        colorize_tree(tree_struct, galah_cannon_subset, abund, path='tree_abund_'+abund+'.png')
        colorize_tree_branches(tree_struct, galah_cannon_subset, abund, path='tree_abund_'+abund+'_branches.png')
    os.chdir('..')

# ------------------------------------------------------------------
# -------------------- Tree analysis begin -------------------------
# ------------------------------------------------------------------
if not perfrom_data_analysis:
    raise SystemExit

from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleStaeckel, estimateDeltaStaeckel

# join datasets
galah_tgas = join(galah_cannon_subset, galah_tgas_xmatch, keys='sobject_id', join_type='inner')

# remove units as they are later assigned to every attribute
galah_tgas['rv_guess'].unit = ''
print 'Creating orbits and computing orbital information'
# create new field in galah-tgas set
for object in galah_tgas:

    # create orbit object
    orbit = Orbit(vxvv=[object['ra_gaia'] * un.deg,
                        object['dec_gaia'] * un.deg,
                        1./object['parallax'] * un.kpc,
                        object['pmra'] * un.mas/un.yr,
                        object['pmdec'] * un.mas/un.yr,
                        object['rv_guess'] * un.km/un.s], radec=True)
    ts = np.linspace(0, 15., 100) * un.Gyr
    ts2 = np.linspace(10, 15., 25) * un.Gyr
    ts3 = np.linspace(5, 10., 25) * un.Gyr
    orbit.integrate(ts, MWPotential2014)
    d1 = estimateDeltaStaeckel(MWPotential2014, orbit.R(ts), orbit.z(ts))
    orbit.integrate(ts2, MWPotential2014)
    d2 = estimateDeltaStaeckel(MWPotential2014, orbit.R(ts2), orbit.z(ts2))
    orbit.integrate(ts3, MWPotential2014)
    d3 = estimateDeltaStaeckel(MWPotential2014, orbit.R(ts3), orbit.z(ts3))
    print d1, d2, d3

