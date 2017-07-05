from NJ_tree_analysis_functions import *
from colorize_tree import *

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir, get_abundance_cols

input_dir = 'NJ_tree_cannon_1.2_mainrun_abundflags_chi2_prob_tgas_norep_norm_megacc_manhattan'
nwm_tree_file = 'distances_network.nwk'

move_to_dir(input_dir)

tree_struct = get_tree_from_file(nwm_tree_file)
# start tree gui
mark_objects(tree_struct, [], path=None, min_mark=0)



