from __future__ import print_function
import os
import re
import numpy as np
import pdb

#prefix = 'result/actfast/'
prefix = 'result/mimic3/'
seed_range = 5
n_groups = 4
res_dir_bases = []

res_dir_bases += ['HAP']  # HAP
res_dir_bases += ['HAP_lv3']  # HAP
res_dir_bases += ['HAP_lv2']  # HAP
res_dir_bases += ['GRAM']  # Gram
res_dir_bases += ['GRAM_lv3']  # Gram
res_dir_bases += ['GRAM_lv2']  # Gram
res_dir_bases += ['RNN_plus']  # RNN+
res_dir_bases += ['RNN']  # RNN
res_dir_bases += ['Rollup_plus']  # Rollup+
res_dir_bases += ['Rollup']  # Rollup

for res_dir_base in res_dir_bases:
    grouped_acc = np.zeros([seed_range, n_groups])
    acc = np.zeros(seed_range)
    for seed in range(seed_range):
        res_dir = os.path.join(prefix + res_dir_base + '_s' + str(seed) + '/', 'result.log')
        with open(res_dir, 'r') as f:
            for line in f:
                pass
            res = [x for x in re.split(', |:|\[|\]|\n', line) if x]
            grouped_acc[seed, :] = [float(x) for x in res[-n_groups:]]
            acc[seed] = float(res[13])
    print('\033[91m Results of ' + res_dir_base + '\033[00m')
    for a, b in zip(np.around(np.mean(grouped_acc, 0), 4), np.around(np.std(grouped_acc, 0), 4)):
        print('%.4f$\pm$%.4f'%(a, b), end =" & ")
    #print()
    #print('Mean and std of test accuracy:')
    print('%.4f$\pm$%.4f'%(np.around(np.mean(acc), 4), np.around(np.std(acc), 4)))
    
