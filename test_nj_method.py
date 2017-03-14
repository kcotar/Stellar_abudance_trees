import os
import numpy as np

from sklearn.metrics.pairwise import manhattan_distances
from skbio import DistanceMatrix
from skbio.tree import nj
from astropy.table import Table

galah_data_dir = '/home/klemen/GALAH_data/'
galah_cannon = Table.read(galah_data_dir+'sobject_iraf_cannon_1.2.fits')[100000:105000]

abund_cols = [col for col in galah_cannon.colnames if '_abund_' in col and '_e_' not in col]
# galah_ids = galah_cannon['sobject_id']._void
galah_cannon_data = np.array(galah_cannon[abund_cols].to_pandas())
print np.shape(galah_cannon_data)

distances = manhattan_distances(galah_cannon_data)
# export distances to be used by megacc procedure
txt = open('distances.meg', 'w')
txt.write('#mega\n')
txt.write('!Title Mega file;\n')
txt.write('!Description Test file;\n')
txt.write('!Format DataType=distance DataFormat=lowerleft;\n')
# export all taxa
for s_id in galah_cannon['sobject_id']:
    txt.write('#'+str(s_id)+'\n')
# output settings
for i_r in range(1, distances.shape[0]):
    txt.write(' '.join([str(f) for f in distances[i_r, 0: i_r]])+'\n')
txt.close()

os.system('megacc -a infer_NJ_distances.mao -d distances.meg -o distances_network.nwk')

# dm = DistanceMatrix(distances)
# nj_tree = nj(dm)
# print(nj_tree.ascii_art())