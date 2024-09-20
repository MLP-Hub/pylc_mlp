
# PyLC Landscape Classifier

__Semantic segmentation for land cover classification of oblique ground-based photography__

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

 Reference: Rose, Spencer, _An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography_,
 MSc. Thesis 2020, University of Victoria.
 <http://hdl.handle.net/1828/12156>

## Overview

The PyLC (Python Landscape Classifier) is a Pytorch-based trainable segmentation network and land cover classification tool for oblique landscape photography. PyLC was developed for the land cover classification of high-resolution grayscale and colour oblique mountain photographs. The training dataset is sampled from the [Mountain Legacy Project](http://mountainlegacy.ca/) repeat photography collection hosted at the [University of Victoria](https://uvic.ca/).

The Deeplab implementation was adapted from [Jianfeng Zhang, Vision & Machine Learning Lab, National University of Singapore, Deeplab V3+ in PyTorch](https://github.com/jfzhang95/pytorch-deeplab-xception). The U-Net implementation was adapted from [Xiao Cheng](https://github.com/xiaochengcike/pytorch-unet-1).

PyTorch implementation of Deeplab: This is a PyTorch(0.4.1) implementation of DeepLab-V3-Plus. It can use Modified Aligned Xception and ResNet as backbone.

### Mountain Legacy Project (MLP)

 The [Mountain Legacy Project](http://mountainlegacy.ca/) supports numerous research initiatives exploring the use of repeat photography to study ecosystem, landscape, and anthropogenic changes. MLP hosts the largest systematic collection of mountain photographs, with over 120,000 high-resolution historic (grayscale) survey photographs of Canada’s Western mountains captured from the 1880s through the 1950s, with over 9,000 corresponding modern (colour) repeat images. Over the years, the MLP has built a suite of custom tools for the classification and analysis of images in the collection (Gat et al. 2011; Jean et al. 2015b; Sanseverino et al. 2016).

### Implementation

PyLC uses deep convolutional neural networks (DCNNs) trained on high-resolution, grayscale and colour landscape photography from the MLP collection, specifically optimized for the segmentation of oblique mountain landscapes. This package uses [U-net](ref-3) and [Deeplabv3+](#ref-4) segmentation models with a ResNet101 pretrained encoder, as well as a fully-connected [conditional random fields model](#ref-5) used to boost segmentation accuracy.

### Features

- Allows for classification of high-resolution oblique landscape images
- Uses multi-loss (weighted cross-entropy, Dice, Focal) to address semantic class imbalance
- Uses threshold-based data augmentation designed to improve classification of low-frequency classes
- Applies optional Conditional Random Fields (CRF) filter to boost segmentation accuracy

## Requirements (Python 3.10+)

All DCNN models and preprocessing utilities are implemented in [PyTorch](https://pytorch.org/), an open source Python library based on the Torch library and [OpenCV](https://opencv.org/), a library of programming functions developed for computer vision. Dependencies are listed below.

- [numpy](https://numpy.org/) >=1.18.5
- [h5py](https://www.h5py.org/) >= 2.8.0
- [opencv-contrib-python](https://opencv.org/) >= 4.10.0.84
- [torch](https://pytorch.org/) >=1.6.0
- [seaborn](https://seaborn.pydata.org/) >=0.11.0(optional - evaluation)
- [matplotlib](https://matplotlib.org/) >=3.2.2 (optional - evaluation)
- [scikit-learn](https://scikit-learn.org/stable/) >=0.23.1(optional - evaluation)
- [tqdm](https://github.com/tqdm/tqdm) >=4.47.1

## Usage

The PyLC (Python Landscape Classifier) classification tool has three main run modes:

1. Data Preprocessing;
 - Extraction: To generate training and validation databases.
 - Profiling: To profile the semantic class distribution of a dataset.
 - Data Augmentation: To extend dataset.
2. Model Training: Train or retrain segmentation networks.
3. Model Testing: Generate segmentation maps.

Default parameters are defined in `config.py`. Categorization schemas (i.e. class labels) are defined in separate JSON files in the local `/schemas` folder. Two examples are provided: `schema_a.json` and `schema_anthro.json`, which adds the anthropogenic class.

### 1. Preprocessing

This package offers configurable preprocessing utilities to prepare raw input data for model training. Input images must be either JPG or TIF format, and masks PNG format. The image filename must match its mask filename (e.g. img_01.tif and msk_01.png). You can download the original image/mask dataset(s) (see repository links under Datasets section) and save to a local directory for model training and testing.

#### 1.1 Extraction

Extraction is a preprocessing step to create usable data to train segmentation network models. Tile extraction is used to partition raw high-resolution source images and masks into smaller square image tiles that can be used in memory. Images are by default scaled by factors of 0.2, 0.5 and 1.0 before tiling to improve scale invariance. Image data is saved to HDF5 binary data format for fast loading. Mask data is also profiled for analysis and data augmentation. See parameters for dimensions and stride. Extracted tiles can be augmented using data augmentation processing.

To create an extraction database from raw images and masks, provide separacte images and masks directory paths. Each image file in the directory must have a corresponding mask file that shares the same file name and use allowed image formats (see above).

Note that the generated database file is saved to `data/db/` in the project root.

##### Options:

- `--img <path>`: (Required) Path to images directory.
- `--mask <path>`: (Required) Path to masks directory.
- `--schema <path>`: (Default: `./schemas/schema_a.json`) Path to JSON categorization schema file.
- `--ch <int>`: (Required) Number of image channels. RGB: 3 (default), Grayscale: 1.
- `--scale <int>`: Apply image scaling before extraction.

```
% python pylc.py extract --ch [number of channels] --img [path/to/image(s)] --mask [path/to/mask(s)]
```

#### 1.3 Data Augmentation

Data augmentation can improve the balance of pixel class distribution in a database by extending the dataset with altered copies of samples composed of less-represented semantic classes. This package uses a novel self-optimizing thresholding algorithm applied to the class distribution of each tile to compute a sampling rate for that tile.

Note that the generated augmented database is saved to `data/db/` in the project root.

##### Options:

- `--db <path>`: (Required) Path to source database file.

```
% python pylc.py augment --db [path/to/database.h5]
```

### 2.0 Training

Training or retraining a model requires an extraction or augmented database generated using the preprocessing steps above. Model training is CUDA-enabled. Note that other training hyperparamters can be set in the `config.py` configuration file. Note that files generated for best models and checkpoints (`.pth`), as well as loss logs (`.npy`), are saved to `./data/save/` in a folder labeled by the model ID.


##### Options:

- `--db <path>`: (Required) Path to training database file.
- `--batch_size <int>`: (Default: 8) Size of each data batch (default: 8).
- `--use_pretrained <bool>`: (Default: True) Use pretrained model to initialize network parameters.
- `--arch [deeplab|unet]`: (Default: 'deeplab') Network architecture.
- `--backbone [resnet|xception]`: (Default: 'resnet') Network model encoder (Deeplab).
- `--weighted <bool>`: (Default: 'True') Weight applied to classes in loss computations.
- `--ce_weight <float>`: (Default: 0.5) Weight applied to cross-entropy losses for back-propagation.
- `--dice_weight <float>`: (Default: 0.5) Weight applied to Dice losses for back-propagation.
- `--focal_weight <flost>`: (Default: 0.5) Weight applied to Focal losses for back-propagation.
- `--optim [adam, sgd]`: (Default: 'adm) Network model optimizer.
- `--sched [step_lr, cyclic_lr, anneal]`: (Default: 'step_lr') Network model optimizer.
- `--normalize`: (Default: 'batch') Network layer normalizer.
- `--activation [relu, lrelu, selu, synbatch]`: (Default: 'relu') Network activation function.
- `--up_mode ['upconv', 'upsample']`: (Default: 'upsample') Interpolation for upsampling (Optional: use for U-Net).
- `--lr <float>`: (Default: 0.0001) Initial learning rate.
- `--batch_size <int>`: (Default: 8) Size of each training batch.
- `--n_epochs <int>`: (Default: 20) Number of epochs to train.
- `--pretrained <bool>`: (Default: True) Use pre-trained network weights (model path defined in `config.py`).
- `--n_workers <int>`: (Default: 6) Number of workers for worker pool.
- `--report`: Report interval (number of iterations).
- `--resume <bool>`: (Default: True) Resume training from existing checkpoint.
- `--clip <float>`: (Default: 1.0) Fraction of dataset to use in training.

```
% python pylc.py train  --db [path/to/database.h5]
```

### 3.0 Testing
Segmentation masks are be generated for input images. Evaluation metrics can also be computed if ground truth masks are provided. Note that image pixel normalization coefficients are stored in model metadata.

##### Options:
- `--model <path>`: (Required) Path to trained model.
- `--img <path>`: (Required) Path to images directory or single file.
- `--mask <path>`: (Optional) Path to masks directory or single file. This option triggers an evaluation of model outputs using various metrics: F1, mIoU, Matthew's Correlation Coefficient, and generates a confusion matrix.
- `--scale <float>`: (Default: 1.0) Scale the input image(s) by given factor.
- `--save_logits <bool>`: (Default: False) Save unnormalized model output(s) to file.
- `--save_probs <bool>`: (Default: False) Save normalized model output(s) to file.
- `--aggregate_metrics <bool>`: (Default: False) Report aggregate metrics for batched evaluations.

```
% python pylc.py test --model [path/to/model] --img [path/to/images(s)] --mask [path/to/mask(s)]
```

## SLURM Usage (Digital Resource Alliance of Canada)
The `scripts_slurm` directory contains scripts to run PyLC using the SLURM batch management software used by Digital Resource Alliance of Canada. Within each script the corresponding python execution is called, with the same options (file paths, etc.) as the usage description above. To us a script first modify the contents to specify the desired options in the python call. The scripts should be run one at a time (each one depends on output from the previous) in the same order as the usage above, namely:
- `sbatch extract_slurm.sh`
- `sbatch augment_slurm.sh`
- `sbatch train_slurm.sh`
- `sbatch test_slurm.sh`

## References

[1]<a name="ref-1"></a> Jean, Frederic, Alexandra Branzan Albu, David Capson, Eric Higgs, Jason T. Fisher, and Brian M. Starzomski. "The mountain habitats segmentation and change detection dataset." In 2015 IEEE Winter Conference on Applications of Computer Vision, pp. 603-609. IEEE, 2015.

[2]<a name="ref-2"></a> Julie Fortin. Landscape and biodiversity change in the Willmore Wilderness Park through Repeat Photography. MSc thesis, University of Victoria, 2015.

[3]<a name="ref-3"></a> Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9351:234–241, 2015. ISSN 16113349. doi: 10.1007/ 978-3-319-24574-4 28. (http://lmb.informatik.uni-freiburg.de/).

[4]<a name="ref-4"></a> Liang Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4):834–848, 2018. ISSN 01628828. doi: 10.1109/TPAMI.2017.2699184.

[5]<a name="ref-5"></a> Philipp Krähenbühl and Vladlen Koltun. Parameter learning and convergent infer- ence for dense random fields. 30th International Conference on Machine Learning, ICML 2013, 28(PART 2):1550–1558, 2013.
