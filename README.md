# CheXzero-tests


## Description

### overview

This project aims to test different implementations of models against the MMIC-CXR dataset to compare the performance, mainly CLIP based architecture using ViT/B-32, ResNet50, SwinTransformer and BioMedicalBERT.


### how to use
Download the project files and and install the dependancies by running "python -m pip install -r requirements.txt". Download the MMIC-CXR datastet from https://physionet.org/content/mimic-cxr-jpg/2.0.0/ and run run_preprocess.py and make sure you add the paths to where you downloaded the dataset. For the validation datset download the CheXpert set from https://aimi.stanford.edu/chexpert-chest-x-rays and get the validation set. For the test set download it from from https://github.com/rajpurkarlab/cheXpert-test-set-labels and put the .h5 and csv files from both in the test_data directory

To run the standard CLIP text enconder with ViT or RN edit the model name in train.py at line 149 to either "ViT-B/32" or "RN50". In train.py make sure the embed dimension is 512 for ViT or 1024 for Resnet

For the Swin transformer make sure CLIP is imported from "swin_model" in train.py and run the run_swint.py file, ensure the embed dimension is 768

To run Swin with bioclinicalbert run the run_swint.py file with the flag --use_biobert.

For FLAVA run the run_flava.py file with an embed dimension of 768.

For FLAVA with bioclinicalbert use the --use_biobert flag with FLAVA.

To evaluate model checkpionts you can use the individual run_zeroshot files or just use the universal ensemble. To do so put the model checkpoints in the matching directory in best_models and run zero_shot_ensemble.py. This can also be used to ensemble different model checkpoints together as long as they are in the correct subdirectory in best_models.

### disclaimer

Not all of the code in this project was created by me it is an adaptation of the original chexzero implementation from https://github.com/rajpurkarlab/CheXzero with some further contibutions from my acedemic supervisor

## Getting Started

### Dependencies

* you will need the MMIC-CXR, CheXpert and PadChest datsets
* make sure the requirements.txt is installed
* this program was run on windows 11
