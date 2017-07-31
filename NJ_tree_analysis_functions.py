import os, imp, time
import numpy as np
import astropy.units as un
import astropy.coordinates as coord
import gala.coordinates as gal_coord

from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from ete3 import Tree

imp.load_source('veltrans', '../tSNE_test/velocity_transform.py')
from veltrans import *
imp.load_source('vectorcalc', '../Aquarius_membership/vector_plane_calculations.py')
from vectorcalc import *


def get_tree_from_file(output_nwm_file):
    if os.path.isfile(output_nwm_file):
        txt = open(output_nwm_file, 'r')
        file_str = ''
        for line in txt:
            if line[-1] == ';':
                file_str += line
            else:
                file_str += line[:-1]
        tree_struct = Tree(file_str)
        txt.close()
        return tree_struct
    else:
        print 'ERROR: check selected algorithm as it did not produce the following file ' + output_nwm_file
        raise SystemExit


def objects_in_list(objects, o_list):
    idx_in = np.in1d(o_list, objects)
    return np.sum(idx_in)


def get_decendat_sobjects(node):
    desc_names = list([])
    for desc in node.get_descendants():
        name_t = desc.name
        if name_t != '':
            desc_names.append(desc.name)
    return np.sort(desc_names)


def get_decendat_names(node):
    desc_names = list([])
    for desc in node.get_descendants():
        desc_names.append(desc.name)
    return np.sort(desc_names)


def get_children_names(node):
    desc_names = list([])
    for desc in node._get_children():
        desc_names.append(desc.name)
    return np.sort(desc_names)


def is_node_before_leaves(node, min_leaves=2):
    desc_names = get_children_names(node)
    if len(desc_names) < min_leaves:
        return False
    else:
        return (np.array(desc_names) != '').all()


def get_data_subset(data, ids, select_by='sobject_id'):
    idx_galah_tgas = np.where(np.in1d(data[select_by], np.int64(ids)))
    return data[idx_galah_tgas]


def evaluate_pairwise_distances(intersect, median=True, meassure='manh'):
    # compute all possible distances between stellar vectors that
    if meassure is 'manh':
        intersect_dist = manhattan_distances(intersect)
    elif meassure is 'eucl':
        intersect_dist = euclidean_distances(intersect)
    idx_uniq_dist = np.tril_indices_from(intersect_dist, k=-1)
    if median:
        return np.median(intersect_dist[idx_uniq_dist])
    else:
        return intersect_dist[idx_uniq_dist].ravel()


def evaluate_angles(angles):
    return np.median(angles)


def predict_stream_description(data, xyz_out=False, vel_pred=None):
    stars_coord = coord.SkyCoord(ra=data['ra_gaia'] * un.deg,
                                 dec=data['dec_gaia'] * un.deg,
                                 distance=1e3 / data['parallax'] * un.pc)
    xyz_gal = stars_coord.transform_to(coord.Galactocentric)
    xyz_gal_pos = np.array(np.vstack((xyz_gal.x, xyz_gal.y, xyz_gal.z)).T)
    # convert to cylindrical representation
    xyz_gal.representation = 'cylindrical'
    rpz_gal_pos = np.array(np.vstack((xyz_gal.rho, xyz_gal.phi, xyz_gal.z)).T)
    pm_stack = np.vstack((data['pmra']._data, data['pmdec']._data))*un.mas/un.yr
    uvw_gal_vel = gal_coord.vhel_to_gal(stars_coord, pm=pm_stack, rv=data['rv_guess']._data*un.km/un.s).T

    if vel_pred is None:
        # determine stream direction as mean velocity vector
        vel_pred = np.median(uvw_gal_vel, axis=0)

    return xyz_gal_pos, rpz_gal_pos, uvw_gal_vel

    # # calculate points of intersection between plane perpendicular to mean of stellar velocity vectors
    # stream_plane_intersects = stream_plane_vector_intersect(xyz_pos, xyz_vel, stream_pred)
    # stream_plane_angles = stream_plane_vector_angle(xyz_vel, stream_pred)
    # # return computed values
    # if xyz_out:
    #     return stream_pred, stream_plane_angles, stream_plane_intersects, xyz_pos, xyz_vel
    # else:
    #     return stream_pred, stream_plane_angles, stream_plane_intersects


def start_gui_explorer(objs, manual=True, save_dir='', i_seq=1, kinematics_source='', initial_only=False):
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
        if i_seq is not None:
            exec_str += out_file + ' ' + save_dir+'/node_{:04d}'.format(i_seq)
        else:
            exec_str += out_file + ' ' + save_dir
    # add kinematics use information
    exec_str += ' '+kinematics_source

    if not manual:
        if initial_only:
            exec_str += ' True'
        else:
            exec_str += ' False'

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