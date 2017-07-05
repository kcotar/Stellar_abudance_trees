import numpy as np

from ete3 import Tree, TreeStyle, NodeStyle, CircleFace, faces
from colour import Color
from NJ_tree_analysis_functions import *



def mark_objects(tree, targets, path='tree.png', min_mark=1):
    # create a copy of three that will be modified later on
    tree_copy = tree.copy('newick')
    # define tree plot style
    tree_plot = TreeStyle()
    tree_plot.show_leaf_name = False
    tree_plot.show_branch_length = False
    tree_plot.show_branch_support = False
    tree_plot.mode = 'c'
    # also count number of marks
    n_marks = 0
    for branch in tree_copy.traverse():
        b_style = NodeStyle()
        b_style['size'] = 0
        if branch.is_leaf():
            branch_sobject_id = np.int64(branch.name)
            if branch_sobject_id in targets:
                b_style['hz_line_color'] = 'red'
                b_style['hz_line_width'] = 5
                b_facet = CircleFace(radius=10, color='red', style='circle')
                branch.add_face(b_facet, 0, position='branch-right')
                n_marks += 1
        branch.set_style(b_style)
    if n_marks >= min_mark:
        print '  marked: '+str(n_marks)
        if path is None:
            tree.show(tree_style=tree_plot)
        else:
            tree_copy.render(path, h=8000, w=8000, units='px', tree_style=tree_plot)


def colorize_tree(tree, dataset, feature, path='tree.png'):
    # create a copy of three that will be modified later on
    tree_copy = tree.copy('newick')
    # define tree plot style
    tree_plot = TreeStyle()
    tree_plot.show_leaf_name = False
    tree_plot.show_branch_length = False
    tree_plot.show_branch_support = False
    tree_plot.mode = 'c'
    # determine colour set for given dataset
    colorize_dataset = dataset[feature]
    n_steps = 300
    color_labels = list(Color('red').range_to(Color('blue'), n_steps))
    colour_data_min = np.nanpercentile(colorize_dataset, 1.)
    colour_data_max = np.nanpercentile(colorize_dataset, 99.)
    colour_data_range = colour_data_min + np.arange(0, n_steps) * (colour_data_max - colour_data_min) / n_steps
    # print colour_data_min, colour_data_max
    # print colour_data_range[0], colour_data_range[-1]
    # iterate over every tree element
    for branch in tree_copy.traverse():
        if branch.is_leaf():
            idx = np.where(dataset['sobject_id'] == np.int64(branch.name))
            data_val = colorize_dataset[idx]
            if np.isfinite(data_val):
                colour_string = str(color_labels[np.nanargmin(np.abs(colour_data_range - data_val))])
                # node style
                # b_style = NodeStyle()
                # b_style['fgcolor'] = colour_string
                # b_style['size'] = 0
                # b_style['hz_line_color'] = colour_string
                # b_style['hz_line_width'] = 40
                # branch.set_style(b_style)
                b_facet = CircleFace(radius=10, color=colour_string, style='circle')
                branch.add_face(b_facet, 0, position='float')
        else:
            b_style = NodeStyle()
            b_style['size'] = 0
            branch.set_style(b_style)
    if path is None:
        tree.show(tree_style=tree_plot)
    else:
        tree_copy.render(path, h=8000, w=8000, units='px', tree_style=tree_plot)


def colorize_tree_branches(tree, dataset, feature, path='tree_branches.png', leaves_only=False):
    # create a copy of three that will be modified later on
    tree_copy = tree.copy('newick')
    # define tree plot style
    tree_plot = TreeStyle()
    tree_plot.show_leaf_name = False
    tree_plot.show_branch_length = False
    tree_plot.show_branch_support = False
    tree_plot.mode = 'c'
    # determine colour set for given dataset
    colorize_dataset = dataset[feature]
    n_steps = 300
    color_labels = list(Color('red').range_to(Color('blue'), n_steps))
    colour_data_min = np.nanpercentile(colorize_dataset, 1.)
    colour_data_max = np.nanpercentile(colorize_dataset, 99.)
    colour_data_range = colour_data_min + np.arange(0, n_steps) * (colour_data_max - colour_data_min) / n_steps
    # traverse the tree and colorize the tree branches along traversing
    for branch in tree_copy.traverse():
        if leaves_only:
            if not branch.is_leaf():
                continue
        leaves = np.int64(branch.get_leaf_names())
        idx_leaves = np.in1d(dataset['sobject_id'], leaves, assume_unique=True, invert=False)
        if np.sum(idx_leaves) == 0:
            # none of the leaves was found in the original dataset or something went wrong
            continue
        data_val_mean = np.nanmean(colorize_dataset[idx_leaves])
        colour_string = str(color_labels[np.nanargmin(np.abs(colour_data_range - data_val_mean))])
        # node/branch style
        b_style = NodeStyle()
        b_style['size'] = 0
        b_style['hz_line_color'] = colour_string
        b_style['hz_line_width'] = 40
        b_style['vt_line_color'] = colour_string
        b_style['vt_line_width'] = 40
        branch.set_style(b_style)
    if path is None:
        tree.show(tree_style=tree_plot)
    else:
        tree_copy.render(path, h=8000, w=8000, units='px', tree_style=tree_plot)
