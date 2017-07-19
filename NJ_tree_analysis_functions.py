import os, imp
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


def evaluate_pairwise_distances(intersect, median=True, meassure = 'manh'):
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
