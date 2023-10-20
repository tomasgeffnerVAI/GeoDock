import os
import esm
import time
import math
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
from einops import rearrange, repeat
# from src.utils.pdb import save_PDB, place_fourth_atom 
from geodock.utils.pdb import save_PDB, place_fourth_atom 
from torch_geometric.data import HeteroData
# from esm.inverse_folding.util import load_coords  # This was the original one that fails for "CSO"
# from geodock.utils.esm_utils_struct import load_coords  # I don't get why this does not work wtffff

import sys
sys.path.append("/home/tomasgeffner/GeoDock/geodock/utils")
from esm_utils_struct import load_coords 

class GeoDockDataset(data.Dataset):
    def __init__(
        self, 
        dataset: str = 'pinder',
        device: str = 'cuda',
    ):
        # if dataset == 'dips_test':
        #     self.data_dir = "/home/tomasgeffner/GeoDock/geodock/data/test"
        #     self.file_list = [i[:-21] for i in os.listdir(self.data_dir) if i[-3:] == 'pdb'] 
        #     self.file_list = list(dict.fromkeys(self.file_list))  # remove duplicates
        #     print(self.file_list)
        #     exit()

        if dataset == 'pinder':
            self.data_dir = "/home/tomasgeffner/pinder_copy"

            file_list = []
            for root, dirs, files in os.walk(self.data_dir):
                for f in files:
                    if ".pdb" in f:
                        file_list.append(os.path.join(root, f))
            self.file_list = file_list
        
        self.dataset = dataset
        self.device = device
        self.fail_list = []

        # Load esm
        # This to download: model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet('/home/tomasgeffner/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
        self.batch_converter = alphabet.get_batch_converter()
        self.esm_model = esm_model.to(device).eval()

    def __getitem__(self, idx: int):
        # if self.dataset == 'dips_test':
        #     _id = self.file_list[idx] 
        #     pdb_file_1 = os.path.join(self.data_dir, _id+".dill_r_b_COMPLEX.pdb")
        #     pdb_file_2 = os.path.join(self.data_dir, _id+".dill_l_b_COMPLEX.pdb")
        #     coords1, seq1 = load_coords(pdb_file_1, chain=None)
        #     coords2, seq2 = load_coords(pdb_file_2, chain=None)
        #     coords1 = torch.nan_to_num(torch.from_numpy(coords1))
        #     coords2 = torch.nan_to_num(torch.from_numpy(coords2))
        
        fail = False

        if self.dataset == 'pinder':
            pdb_file = self.file_list[idx]
            # coords, seq = load_coords(pdb_file, chain=None)
            try:
                # This line is sometimes problematic leads to some failures
                coords, seq = load_coords(pdb_file, chain=None)
                # "structure has multiple atoms with the same name" comes from
                #   https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/inverse_folding/util.py#L105C35-L105C35
                # For isntance, for this file /home/tomasgeffner/pinder_copy/splits/test/1jma__A1_Q92956--1jma__B1_P57083/apo/1jma__B1_P57083_L.pdb

                # Update, this has to do with broken APOs, and a few others...

                # Other error is "CSO" is a key error comes from ProteinSequence._dict_3to1[symbol.upper()] (I think symbol.upper is "CSO") comes from
                #   https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/inverse_folding/util.py#L73C1-L73C1
                # For isntance, for this file /home/tomasgeffner/pinder_copy/splits/test/2es4__A1_P0DUB8--2es4__B1_Q05490/2es4__A1_P0DUB8--2es4__B1_Q05490.pdb

                # This has to do with non-standard aa
                # Fixed by changing the function used to transform non standard to standard variants

                # Found
                # "CSO" -> "CYS" 
                # "SEP" -> "SER"
                # "TPO" -> "THR"
                # "MLY" -> "LYS"
            
            except Exception as e:
                fail = True
                print("fail", len(self.fail_list))
                print(pdb_file)
                print(e)
                self.fail_list.append(pdb_file)
            
        if not fail:
            coords = torch.nan_to_num(torch.from_numpy(coords))

            # ESM embedding
            esm_rep = self.get_esm_rep(seq)

            # save data to a hetero graph 
            data = HeteroData()

            data['prot'].x = esm_rep
            data['prot'].pos = coords
            data['prot'].seq = seq
            data.name = pdb_file
            assert pdb_file[-4:] == ".pdb"
            out_name = pdb_file[:-4] + ".pt"
            torch.save(data, out_name)

            return coords
        
        return torch.ones(3)

    def __len__(self):
        return len(self.file_list)

    def get_esm_rep(self, seq_prim):
        # Use ESM-1b format.
        # The length of tokens is:
        # L (sequence length) + 2 (start and end tokens)
        seq = [
            ("seq", seq_prim)
        ]
        out = self.batch_converter(seq)
        with torch.no_grad():
            results = self.esm_model(out[-1].to(self.device), repr_layers = [33])
            rep = results["representations"][33].cpu()
        
        return rep[0, 1:-1, :]


if __name__ == '__main__':
    name = 'pinder'
    # name = 'dips_test'

    dataset = GeoDockDataset(
        dataset=name,
        device="cuda",
    )

    dataloader = data.DataLoader(dataset, batch_size=1)

    count = 0
    for batch in tqdm(dataloader):
        count += 1
        # if count > 100:
        #     break

    print(len(dataset.fail_list), len(dataloader))

