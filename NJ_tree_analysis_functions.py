import imp
import numpy as np
import astropy.units as un
import astropy.coordinates as coord

from sklearn.metrics.pairwise import manhattan_distances

imp.load_source('veltrans', '../tSNE_test/velocity_transform.py')
from veltrans import *
imp.load_source('vectorcalc', '../Aquarius_membership/vector_plane_calculations.py')
from vectorcalc import *


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


def evaluate_pairwise_distances(intersect, median=True):
    # compute all possible distances between stellar vectors that
    intersect_dist = manhattan_distances(intersect)
    idx_uniq_dist = np.tril_indices_from(intersect_dist, k=-1)
    if median:
        return np.median(intersect_dist[idx_uniq_dist])
    else:
        return intersect_dist[idx_uniq_dist].ravel()


def evaluate_angles(angles):
    return np.median(angles)


def predict_stream_description(data, xyz_out=False, stream_pred=None):
    xyz_pos = coord.SkyCoord(ra=data['ra_gaia'] * un.deg,
                             dec=data['dec_gaia'] * un.deg,
                             distance=1e3 / data['parallax'] * un.pc).cartesian
    xyz_vel = motion_to_cartesic(data['ra_gaia'], data['dec_gaia'],
                                 data['pmra'], data['pmdec'],
                                 data['rv_guess'], plx=data['parallax']).T
    if stream_pred is None:
        # determine stream direction as mean velocity vector
        stream_pred = mean_velocity(xyz_vel)
    # calculate points of intersection between plane perpendicular to mean of stellar velocity vectors
    stream_plane_intersects = stream_plane_vector_intersect(xyz_pos, xyz_vel, stream_pred)
    stream_plane_angles = stream_plane_vector_angle(xyz_vel, stream_pred)
    # return computed values
    if xyz_out:
        return stream_pred, stream_plane_angles, stream_plane_intersects, xyz_pos, xyz_vel
    else:
        return stream_pred, stream_plane_angles, stream_plane_intersects
