# PTrans-ODON

## Structure

```
.
├── IN/
│   ├── H5PY/
│   │   └── dataset_h5py/
│   ├── PNG/
│   │   └── dataset_png/
│   └── labels.csv
├── OUT/
│   ├── data_aug_IP/
│   ├── data_aug_mix/
│   ├── data_aug_total_IP/
│   └── data_aug_treatments/
├── tools/
│   ├── IPDL/
│   │   ├── Image_Inpainting_Autoencoder.ipynb
│   │   ├── inference.py
│   │   └── model_weights.keras
│   ├── treatments_generation/
│   │   └── treatments_generation.py
│   ├── dataset_annotation/
│   │   ├── OUT/
│   │   ├── find_treatments.ipynb
│   │   └── font.ttf
│   └── segmentation_assessment/
│       ├── assess_data/
│       │   └── scores.csv
│       └── src/
│           └── dataset_eval.py
├── .gitignore
├── gen_aug_IP.py
├── gen_aug_mix.py
├── gen_aug_treatments.py
└── README.md
```

## Requirements

Below are the packages required to run the project:
- h5py==3.6.0
- numpy==1.22.4
- matplotlib=3.5.2
- opencv-python==4.5.5.64
- scikit-image==0.19.2
- tensorflow==2.9.1


## Usage
### Data augmentation
1. Add the .h5py images dataset to `./IN/H5PY/h5py_dataset/` 
2. Run a combination of `gen_aug_IP.py`,  `gen_aug_treatments.py` and `gen_aug_mix.py`
