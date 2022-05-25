import h5py
from glob import glob
import random
import csv

import numpy as np
import cv2
import os

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from tools.IPDL import inference


print("""
===========================================
=                                         =
=   Data augmentation with Inpainting     =
=                                         =
===========================================
""")

model = inference.createModel("tools/IPDL/model_weights.keras")

label_file = "IN/labels.csv"

h5py_files = glob('IN/H5PY/dataset_h5py/*.h5py')

treatments = {}
with open(label_file, 'r') as f:
    file = csv.reader(f, delimiter=',')
    for line in file:
        treatments[line[0]] = {}
        for i in range(1, 33):
            treatments[line[0]][i] = bool(int(line[i][-1]))

kernel = np.ones((5,5))

progress_index = 0
progress_max = len(h5py_files)

for p in h5py_files:
    with h5py.File(p) as f:
        img = cv2.resize(np.array(f['X'])[...,0], (1024, 512), cv2.INTER_NEAREST)/255
        name = os.path.splitext(os.path.basename(p))[0]
        
        tooth_t = [t for t in treatments[name] if treatments[name][t]]
        tooth_inpaint = []

        inpaint = inference.infer(model, img)

        for i in tooth_t:
            segm = cv2.resize(np.array(f['y'])[...,i], (1024, 512), cv2.INTER_NEAREST)
            segm = cv2.dilate(segm, kernel, iterations=1)
            tooth_inpaint.append(np.multiply(segm, inpaint))

        smartinpaint = img
        for tooth in tooth_inpaint:
            smartinpaint = np.where(tooth, tooth, smartinpaint) 
            
        for i in range(len(tooth_inpaint)+1):
            inpaint_i = img
            selected_treatments = random.sample(tooth_inpaint, i)
            for tooth in selected_treatments:
                inpaint_i = np.where(tooth, tooth, inpaint_i)
            plt.imsave("OUT/data_aug_IP/" + name + "_" + str(i) + ".png", inpaint_i, cmap='gray')

            
                
        plt.imsave("OUT/data_aug_total_IP/" + name + ".png", smartinpaint, cmap='gray')

        progress_index += 1
        print(f"Progress : {round(100*progress_index/progress_max, 1)}%")