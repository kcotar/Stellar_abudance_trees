import numpy as np
from NJ_tree_analysis_functions import *

tree_folder = '/home/klemen/Stellar_abudance_trees/'
tree_folder += 'NJ_tree_cannon_3.0_abundflags_ucac5_norep_norm_hier_weighted_cityblocknans_loose_ra_80.0_dec_5.0_rad_35.0/'
# find the following sobject_id in a tree and analyze a subtree that it belongs to
sobject_id = 170122002601361

# read constructed classification tree
tree_struct = get_tree_from_file(tree_folder+'distances_network.nwk')

# find sobject id in there if it even exists
target_node = tree_struct.get_leaves_by_name(str(sobject_id))
print target_node

if len(target_node) < 1:
    # only output error message
    print 'Sobject id  '+str(sobject_id)+' not found in tree.'
else:
    # find major split and run the analysis of the subtree
    node_before_object = target_node[0].get_ancestors()[0]
    node_to_analyze = get_major_split(node_before_object, max_add_objects=8, n_levels=8)
    descendants_to_analyze = get_decendat_sobjects(node_to_analyze)
    start_gui_explorer(descendants_to_analyze, manual=True, initial_only=False, loose=True,
                       save_dir=tree_folder, i_seq=1, kinematics_source='ucac5')

