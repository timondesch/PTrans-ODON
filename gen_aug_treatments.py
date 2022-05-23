
import h5py
from glob import glob
import random
import csv

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from tools.treatments_generation.treatments_generation import gen_treatment, fuse


WEIGHTS = [*range(1, 9)]
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)


label_file = "IN/labels.csv"

h5py_files = glob('IN/H5PY/dataset_h5py/*.h5py')


treatments = {}
with open(label_file, 'r') as f:
    file = csv.reader(f, delimiter=',')
    for line in file:
        treatments[line[0]] = {}
        for i in range(1, 33):
            treatments[line[0]][i] = bool(int(line[i][-1]))


for p in h5py_files:
    UPPER = [*range(1, 17)]
    LOWER = [*range(17, 33)]
    with h5py.File(p) as f:
        img = cv2.resize(np.array(f['X'])[...,0], (1024, 512), cv2.INTER_NEAREST)
        name = os.path.splitext(os.path.basename(p))[0]
        
        tooth_t = [t for t in treatments[name] if not treatments[name][t] and np.any(np.array(f['y'])[...,t])]
        weight_t = [((t-1)%8)+1 for t in tooth_t]
        sum_weight_t = sum(weight_t)
        weight_t = [t/sum_weight_t for t in weight_t]

        top = []
        bottom = []
        for i in range(min(6, len(tooth_t)+1)):
            current = img
            selected_teeth = np.random.choice(tooth_t, i, replace=False, p=weight_t)
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


