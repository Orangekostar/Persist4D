# ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes

[Project Page](https://www.easteine.com/rescene4d/) | [arXiv](https://arxiv.org/abs/2601.11508) | [stmetrics](https://github.com/GradientSpaces/stmetrics)

**ReScene4D** is a framework for temporally consistent semantic instance segmentation across evolving indoor 3D scenes. It extends Mask3D-style masked transformers to 4D (multi-timestep) point clouds, and introduces temporal evaluation metrics (t-AP, t-REC) via the companion [stmetrics](https://github.com/GradientSpaces/stmetrics) package.

---

## Code structure

```
├── main_instance_segmentation.py   <- entry point (train + eval)
├── conf                            <- Hydra configuration files
│   ├── config_base_instance_segmentation.yaml
│   ├── backbone/                   <- MinkowskiEngine / Sonata / Concerto backbones
│   ├── data/                       <- dataset configs
│   ├── model/                      <- ReScene model config
│   ├── metrics/                    <- tmap, tsim metric configs
│   └── ...
├── datasets
│   ├── semseg.py                   <- dataset class
│   ├── preprocessing/              <- preprocessing scripts
|   ├── minkowksi_utils             <- minkowski voxelizer 
|   ├── pointcept_utils             <- pointcept voxelizer 
|   └── ...
├── models                          <- model modules
│   ├── rescene.py                  <- ReScene4D model
│   ├── minkowski.py                <- MinkowskiEngine backbone wrapper
│   ├── pointcept.py                <- Pointcept backbone wrapper
│   └── ...
├── trainer
│   └── trainer.py                  <- PyTorch Lightning train loop
├── data
│   ├── processed/                  <- preprocessed datasets
│   └── raw/                        <- raw datasets / test segmentations
└── saved                           <- model checkpoints and logs
```

---

## Installation

### Prerequisites

- Python >= 3.10
- CUDA 12.6
- GCC 11

There are two starting paths depending on which backbone you want to use. Both converge at **step 3**.

### 1. MinkowskiEngine environment (optional)

MinkowskiEngine is only required for the MinkowskiEngine-based backbone. It can be challenging to build for CUDA 12+. We have compiled online fixes and patched headers in our fork — follow the installation instructions at [GradientSpaces/MinkowskiEngine](https://github.com/GradientSpaces/MinkowskiEngine), which also covers HPC cluster setup.

Once MinkowskiEngine is installed into a `Mink12` conda environment, initialize the ReScene environment from it:

```bash
conda create --name rescene --clone Mink12
conda activate rescene
```


### 2. Pointcept-only environment

Use this path if you only want to run Pointcept-based backbones (Sonata, Concerto) and do not need MinkowskiEngine.

```bash
conda create --name rescene python=3.10
conda activate rescene


# adjust to your CUDA version — this example uses CUDA 12.6 + PyTorch 2.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

```

### 3. Continue here from step 1 or 2 

```bash
# sonata and concerto requirements 
pip install spconv-cu126
pip install git+https://github.com/Dao-AILab/flash-attention.git
pip install huggingface_hub timm
conda install addict pandas scipy -c conda-forge


# third party 
mkdir -p third_party
cd third_party

git clone git@github.com:facebookresearch/sonata.git
cd sonata && python setup.py install && cd ..

git clone git@github.com:Pointcept/Concerto.git
cd concerto && python setup.py install && cd ..

# metrics 
git clone https://github.com/GradientSpaces/stmetrics.git
cd stmetrics && pip install -e . && cd ..

cd pointnet2 && pip3 install . && cd .. 

# only needed for preprocessing 
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make
cd ../..

pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
# volumentations must be installed separately due to dependency conflicts
pip install volumentations --no-dependencies
pip3 install -r requirements.txt

```

---

## Data preprocessing

### 3RScan

```bash
python datasets/preprocessing/segment_script.py --dataset=3rscan \
    --data_dir="/path/to/3RScan" \
    --save_dir="data/raw/rio_test_segments" \
    --metadata_file="/path/to/3RScan/3RScan.json"
```

```bash
python -m datasets.preprocessing.RScan_preprocessing preprocess \
    --data_dir="/path/to/3RScan" \
    --save_dir="data/processed/rio" \
    --scannet200=False \
    --n_jobs=8
```

### ScanNet

Similar to Mask3D, we apply Felzenswalb and Huttenlocher's Graph Based Image Segmentation algorithm to preprocess the pointclouds. Refer to the [original repo](https://github.com/ScanNet/ScanNet/tree/master/Segmentator) for details. 

```bash
python datasets/preprocessing/segment_script.py --dataset=scannet \
    --data_dir="/path/to/ScanNet" \
    --save_dir="data/raw/scannet_test_segments" \
    --git_repo="third_party/ScanNet"
```

```bash
python -m datasets.preprocessing.scannet_preprocessing preprocess \
    --data_dir="/path/to/ScanNet" \
    --save_dir="data/processed/scannet" \
    --git_repo="third_party/ScanNet" \
    --scannet200=False \
    --n_jobs=8
```


## Checkpoints

Coming soon. 

---

## Training and inference

```bash
python main_instance_segmentation.py
```

Hydra configs in `conf/` control the full experiment. To run inference only:

```bash
python main_instance_segmentation.py \
    general.checkpoint='checkpoints/rescene4d.ckpt' \
    general.train_mode=False
```

---

## Evaluation metrics

ReScene4D uses [stmetrics](https://github.com/GradientSpaces/stmetrics) for evaluation. Metrics include temporal AP (t-AP), standard spatial only mAP, and per-timestep AP. See the stmetrics README for full documentation of the API and dataset spec format.

---

## Acknowledgements

This codebase builds on [Mask3D](https://github.com/JonasSchult/Mask3D) by Jonas Schult et al. Please cite them as well!

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{steiner2026rescene4d,
      author = {Steiner, Emily and Zheng, Jianhao and Howard-Jenkins, Henry and Xie, Chris and Armeni, Iro},
      title = {ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2026},
}
```