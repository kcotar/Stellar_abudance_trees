import imp

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import *
imp.load_source('helper2', '../tSNE_test/cannon3_functions.py')
from helper2 import *

imp.load_source('distf', '../tSNE_test/distances.py')
from distf import *

from filter_galah import *

galah_data_dir = '/home/klemen/GALAH_data/'
galah_cannon = Table.read(galah_data_dir+'sobject_iraf_iDR2_171103_cannon.fits')
galah_ann = Table.read(galah_data_dir+'galah_abund_ANN_SME3.0.1_stacked_median.fits')

abund_cols = get_abundance_cols3(galah_cannon.colnames)
abund_cols_ann = get_abundance_colsann(galah_ann.colnames)
abund_cols = remove_abundances(abund_cols, what_abund_remove())
abund_cols_ann = remove_abundances(abund_cols_ann, what_abund_remove(), type='ann')
abund_elem = get_element_names(abund_cols)

galah_filter_ok = FILTER(verbose=True)
galah_filter_ok.filter_attribute(galah_cannon, attribute='sobject_id', value=140301000000000, comparator='>')
idx_ok_flags = quality_flagging(galah_cannon, cannon_flag=True, chi2outliers=True, return_idx=True)
galah_filter_ok._merge_ok(idx_ok_flags)
galah_filter_ok.filter_attribute(galah_cannon, attribute='snr_c2_iraf', value=20, comparator='>')
galah_cannon_subset = galah_filter_ok.apply_filter(galah_cannon)

# galah_cannon_subset = cannon_prepare_flagged_table(galah_cannon_subset, abund_cols, norm=True)
#
# for i_col in range(len(abund_cols)):
#     ab_multi = abund_multiplier_c3(abund_elem[i_col])
#     galah_cannon_subset[abund_cols[i_col]] *= ab_multi

repeats_id = 8786494
sobject_ids = galah_cannon_subset[galah_cannon_subset['galah_id'] == repeats_id]['sobject_id']
print sobject_ids.data

# sobject_ids = [170531004801078, 170725002601154, 170725003101107]
sobject_ids = [160522005601074, 160522005601298, 160531004101099, 160531004101187, 170510006801219, 170531004801132, 170725002601181, 170725003601105]

repeats_abund_nans = galah_cannon_subset[np.in1d(galah_cannon_subset['sobject_id'], sobject_ids)][abund_cols]
abund_ok_cols = np.isfinite(repeats_abund_nans.to_pandas().values).all(axis=0)
repeats_abund_valid = repeats_abund_nans[list(np.array(abund_cols)[abund_ok_cols])]

repeats_abund_ann = galah_ann[np.in1d(galah_ann['sobject_id'], sobject_ids)][list(np.array(abund_cols_ann)[abund_ok_cols])]
abund_raw_data_ann = repeats_abund_ann.to_pandas().values

# abund_raw_data = repeats_abund_nans.to_pandas().values
abund_raw_data = repeats_abund_valid.to_pandas().values

print
print cityblock_nans(abund_raw_data, triu=False)
print cityblock_nans(abund_raw_data_ann, triu=False)
# print
# print abund_raw_data[0,:]-abund_raw_data[1,:]
# print abund_raw_data[2,:]-abund_raw_data[1,:]
# print abund_raw_data[0,:]-abund_raw_data[2,:]
print
print np.rot90(abund_raw_data)
print
print np.rot90(abund_raw_data_ann)

