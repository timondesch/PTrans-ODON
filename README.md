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
│   └── inpainting_assessment/
│       ├── assess_data/
│       └── src/
│           └── dataset_eval.py
├── .gitignore
├── gen_aug_IP.py
├── gen_aug_mix.py
├── gen_aug_treatments.py
├── main.py
└── README.md
```

- `IN/` is where lives the inputs of the scripts (H5PY for data augmentation, PNG for inpainting model training, labels.csv for annotation)
- `OUT` is where the output of data augmentation can be found (`data_aug_IP/` for only inpainting, `...mix/` for a mix of inpainting and false treatments, `...total_IP/` for inpainting on the full image and `...treatments/` for false treatment generation)
- In `tools` is all our useful scripts that are used for our data augmentation (`IPDL/` for deep learning-based inpainting, `treatment_generation/` for false treatments generation, `dataset_annotation/` for treatment annotation and `inpainting_assessment/` for inpainting assessment using a comparison of segmentation)
- In the base directory are also the `gen_aug_*` scripts for generating the augmented datasets as well as a `main` script for running all of them

## Requirements

Below are the packages required to run the project:
- h5py==3.6.0
- numpy==1.22.4
- matplotlib=3.5.2
- opencv-python==4.5.5.64
- scikit-image==0.19.2
- tensorflow==2.9.1

Optional:
- scikit-learn==1.1.1 (for `tools/dataset_annotation/find_treatments.ipynb`)


## Usage
### Data augmentation
1. Add the base .h5py images dataset to `./IN/H5PY/h5py_dataset/` 
2. Run a combination of `gen_aug_IP.py`,  `gen_aug_treatments.py` and `gen_aug_mix.py` or all of them with `main.py`

### Training the inpainting model
The current model is already trained with all available data, and thus retraining it might lead to subpar performances. However if need be, it can be retrained with the following steps:
1. Add the training dataset in `IN/PNG/bases/`. There must be one directory for every image with the same name as the image. In each of them, there must be the image as well as all of its segmentations named `imageName_i.png` with i being the tooth number.
2. Run `tools/treatments_generations.py`
3. Run the training notebook (`tools/IPDL/inpainting_training.ipynb`)

### Tool for finding treatments
1. Add the base .h5py images dataset to `./IN/H5PY/h5py_dataset/` 
2. Run `tools/dataset_annotation/find_treatments.ipynb`
3. Images with annotations can then be found under `tools/dataset_annotation/OUT/`
