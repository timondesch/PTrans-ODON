import h5py
from glob import glob
import csv

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from tools.treatments_generation.treatments_generation import gen_treatment, fuse

print("""
===========================================
=                                         =
= Data augmentation with false treatments =
=                                         =
===========================================
""")

WEIGHTS = [*range(1, 9)]
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)

NUMBER_IMAGES = 10

label_file = "IN/labels.csv"

h5py_files = glob('IN/H5PY/dataset_h5py/*.h5py')

UPPER = [*range(1, 17)]
LOWER = [*range(17, 33)]

treatments = {}
with open(label_file, 'r') as f:
    file = csv.reader(f, delimiter=',')
    for line in file:
        treatments[line[0]] = {}
        for i in range(1, 33):
            treatments[line[0]][i] = bool(int(line[i][-1]))


progress_index = 0
progress_max = len(h5py_files) * NUMBER_IMAGES

for p in h5py_files:
    with h5py.File(p) as f:
        img = cv2.resize(np.array(f['X'])[...,0], (1024, 512), cv2.INTER_NEAREST)
        name = os.path.splitext(os.path.basename(p))[0]
        
        tooth_t = [t for t in treatments[name] if not treatments[name][t] and np.any(np.array(f['y'])[...,t])]
        weight_t = [((t-1)%8)+1 for t in tooth_t]
        sum_weight_t = sum(weight_t)
        weight_t = [t/sum_weight_t for t in weight_t]

        for i in range(NUMBER_IMAGES):
            top = []
            bottom = []
            current = img
            temp = int(np.random.normal(4.5, 2))
            while temp<=0 or temp>len(tooth_t):
                temp = int(np.random.normal(4.5, 2))

            selected_teeth = np.random.choice(tooth_t, temp, replace=False, p=weight_t)
            for tooth in selected_teeth:
                segm = cv2.resize(np.array(f['y'])[...,tooth], (1024, 512), cv2.INTER_NEAREST)
                tooth_w_t = np.where(segm, img, segm)
                treatment = gen_treatment(tooth, tooth_w_t)
                if tooth in UPPER:
                    top.append(treatment)
                else:
                    bottom.append(treatment)
            current = fuse(current, top, bottom)
            plt.imsave("OUT/data_aug_treatments/" + name + "_" + str(i) + ".png", current, cmap='gray')
            progress_index += 1
            print(f"Progress : {round(100*progress_index/progress_max, 1)}%")