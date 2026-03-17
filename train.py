import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from PIL import Image
import h5py

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, Lambda

import sys
sys.path.append('../..')

import clip
#CHANGE THIS TO MODEL WHEN RUNNING VIT/RN
#KEEP AS SWIN_MODEL FOR SWIN AND BIOBERT
from swin_model import CLIP, FLAVAWrapper
from simple_tokenizer import SimpleTokenizer

class CXRDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path, txt_path, column='report', size=None, transform=None, pretrained=False):
        super().__init__()
        if size != None: 
            self.img_dset = h5py.File(img_path, 'r')['cxr'][:size]
            self.txt_dset = pd.read_csv(txt_path)[column][:size]
        else: 
            self.img_dset = h5py.File(img_path, 'r')['cxr']
            self.txt_dset = pd.read_csv(txt_path)[column]
        self.transform = transform
        self.pretrained = pretrained
            
    def __len__(self):
        return len(self.txt_dset)
    

    #vitb32
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        txt = self.txt_dset[idx] # python str
        if type(txt) == type(float("nan")): # capture the case of empty "Impression" sections
            txt = " "
        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)
        sample = {'img': img, 'txt': txt }
        
        return sample

def load_data(cxr_filepath, txt_filepath, batch_size=4, column='report', pretrained=False, verbose=False, flava=False): 
    
    dev = "cuda:0" 
    cuda_available = True
    print('Using CUDA.')
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)

    if flava:
        input_resolution = 224
        transform = Compose([
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
            # Scale from CXR range to [0, 1]
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
        ])
        print("Finished image transforms for FLAVA model.")



    elif pretrained: 
        input_resolution = 224
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])
        print('Interpolation Mode: ', InterpolationMode.BICUBIC)
        print("Finished image transforms for pretrained model.")

   

    """else: 
        input_resolution = 320
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ])
        print("Finished image transforms for clip model.")"""
    
    torch_dset = CXRDataset(img_path=cxr_filepath,
                        txt_path=txt_filepath, column=column, transform=transform, pretrained=pretrained)
    
    if verbose: 
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            plt.imshow(sample['img'][0])
            plt.show()
            print(i, sample['img'].size(), sample['txt'])
            if i == 3:
                break
    
    loader_params = {'batch_size':batch_size, 'shuffle': True, 'num_workers': 0}
    data_loader = data.DataLoader(torch_dset, **loader_params)
    return data_loader, device
    
def load_clip(model_path=None, pretrained=False, context_length=77, swin_encoder=False, use_biobert=False, 
              biobert_model="emilyalsentzer/Bio_ClinicalBERT", flava=False):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a model with the CLIP model 
    architecture. 
    
    args: 
        * model_path (optional) - path to model weights that the model
        will be initialized with 
        * pretrained (optional) - if True, will load the pretrained 
        CLIP model
        * context_length (optional) - length of the maximum number of 
        tokens that can be inputted into the CLIP model
    '''

    params = {
        'embed_dim':768, #512 vit 1024 rn50
        'image_resolution': 320, #288 RN 50X4
        'vision_layers': 12,
        'vision_width': 768, #512 vit 1024 rn50
        'vision_patch_size': 16,
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    }
    
    # set device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if flava:
        model = FLAVAWrapper(
            embed_dim=768,
            use_biobert=use_biobert,
            biobert_model=biobert_model
        )
        model.use_flava = True
        print(f"Loaded FLAVA with: "
              f"{'BioClinicalBERT' if use_biobert else 'Standard BERT'}")

    
    elif pretrained: 
        # load clip pre-trained model
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        for param in model.parameters():
            param.data = param.data.float()
        model.use_biobert = False
        print("Loaded in pretrained model.")
    else: 
        # Load custom CLIP with Swin and/or BioBERT
        model = CLIP(
            swin_encoder=swin_encoder,
            use_biobert=use_biobert,
            biobert_model=biobert_model,
            **params
        )
        
        model_desc = []
        if swin_encoder:
            model_desc.append("Swin Transformer (vision)")
        if use_biobert:
            model_desc.append(f"BioClinicalBERT (text)")
        if not model_desc:
            model_desc.append("standard CLIP")
            
        print(f"Loaded CLIP with: {', '.join(model_desc)}")
    
    # if a model_path is provided, load in weights to backbone
    if model_path != None: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model
    
    
def preprocess_text(texts, model):
#     if model.context_length is None: 
#         model = model.module

    # FLAVA path
 if hasattr(model, 'use_flava') and model.use_flava:
        device = next(model.parameters()).device
        tokens = model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=model.context_length,
            return_tensors="pt"
        )
        return {k: v.to(device) for k, v in tokens.items()}

 elif model.use_biobert:
        # Use BERT tokenizer
        device = next(model.parameters()).device
        tokens = model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,  # enough to process the majority of reports
            return_tensors="pt"
        )
        # Move to device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        return tokens
 
 else:
        
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[:model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def make(config, cxr_filepath, txt_filepath, model_path=None): 
    '''
    FUNCTION: make
    ---------------------------------
    This function makes the model, the data loader, loss and optimizer. 
    
    args: 
        * config - dict, configuration of experiment
        * cxr_filepath - string, filepath to chest x-ray images
        * txt_filepath - string, filepath to corresponding text reports
        * model_path - string, filepath to previously trained model
    '''
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=config.batch_size, pretrained=config.pretrained, column=config.column)
    model = load_clip(model_path=model_path, pretrained=config.pretrained, context_length=config.context_length)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    # todo: incorporate - torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    return model, data_loader, device, criterion, optimizer


def train_main(cxr_filepath, txt_filepath, hyperparams, output_path, model_path=None, pretrained=False): 
    '''
    args: 
        * cxr_filpath- str filepath to cxr images
        * txt_filepath- str filepath to text reports
        * hyperparams- dictionary with the following hyperparams:
        `batch_size`, `criterion`, `learning_rate`, `momentum`, `epochs`
        * output_path- str filepath to where the trained model will be saved
        * model_path- str filepath to model that will be used as baseline model for training. 
        If not provided, a model will be trained from scratch
        * pretrained- whether or not the clip model was pretrained with generic images 
    This function is the main train function for CXR-CLIP. 
    '''
    
    # unpack `hyperparams`
    batch_size = hyperparams['batch_size']
    criterion = hyperparams['criterion']
    learning_rate = hyperparams['learning_rate']
    momentum = hyperparams['momentum']
    epochs = hyperparams['epochs']
    
    # load input cxr + report data
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=batch_size, pretrained=pretrained)
    model = load_clip(model_path=model_path, pretrained=pretrained)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    #(model, data_loader, device, criterion, optimizer, epochs, output_path)
    return model