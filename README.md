# PTrans-ODON

## Structure

```
.
├── IN/
│   ├── H5PY/
│   │   └── dataset_h5py/
│   └── PNG/
│       └── dataset_png/
├── OUT/
│   ├── data_aug_IP
│   ├── data_aug_treatments
│   └── data_aug_mix
├── tools/
│   ├── IPDL/
│   │   ├── Image_Inpainting_Autoencoder.ipynb
│   │   ├── inference.py
│   │   └── model_weights.keras
│   └── treatments_generation/
│       └── treatments_generation.py
├── README.md
└── .gitignore
```

## Usage

1. Add the .h5py images dataset to `./IN/H5PY/h5py_dataset/` 
2. 