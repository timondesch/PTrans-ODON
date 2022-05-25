
import h5py
from glob import glob
import random
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

from tools.treatments_generation.treatments_generation import gen_treatment, fuse
from tools.IPDL import inference


WEIGHTS = [*range(1, 9)]
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)


label_file = "IN/labels.csv"
model = inference.createModel("tools/IPDL/model_weights.keras")
h5py_files = glob('IN/H5PY/dataset_h5py/*.h5py')


treatments = {}
with open(label_file, 'r') as f:
    file = csv.reader(f, delimiter=',')
    for line in file:
        treatments[line[0]] = {}
        for i in range(1, 33):
            treatments[line[0]][i] = bool(int(line[i][-1]))


kernel = np.ones((5,5))


for p in h5py_files:
    with h5py.File(p) as f:
        img = cv2.resize(np.array(f['X'])[...,0], (1024, 512), cv2.INTER_NEAREST)
        name = os.path.splitext(os.path.basename(p))[0]
        
        tooth_inpaint = {}

        inpaint = inference.infer(model, img/255)*255
        plt.imsave("t.png", inpaint, cmap="gray")

        for i in [t for t in treatments[name] if treatments[name][t]]:
            segm = cv2.resize(np.array(f['y'])[...,i], (1024, 512), cv2.INTER_NEAREST)
            segm = cv2.dilate(segm, kernel, iterations=1)
            tooth_inpaint[i] = np.multiply(segm, inpaint)
        

        UPPER = [*range(1, 17)]
        LOWER = [*range(17, 33)]
        for k in range(random.randint(4,8)):
            top = []
            bottom = []
            current = img
            for i in range(1,33):
                if(treatments[name][i]):
                    if np.random.choice([True, False], 1, p=[0.7, 0.3]):
                        current = np.where(tooth_inpaint[i], tooth_inpaint[i], current)

                else:
                    if(np.any(np.array(f['y'])[...,i])) and np.random.choice([True, False], 1, p=[(((i-1)%8+1)/18), 1 - (((i-1)%8+1)/18)]):            
                        segm = cv2.resize(np.array(f['y'])[...,i], (1024, 512), cv2.INTER_NEAREST)
                        tooth = np.where(segm, img, segm)
                        treatment = gen_treatment(i, tooth)
                        if i in UPPER:
                            top.append(treatment)
                        else:
                            bottom.append(treatment)
                        
            current = fuse(current, top, bottom)

            plt.imsave("OUT/data_aug_mix/" + name + "_" + str(k) + ".png", current, cmap='gray')