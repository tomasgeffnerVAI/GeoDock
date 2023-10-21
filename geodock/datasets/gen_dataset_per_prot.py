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
from torch_geometric.data import HeteroData

import sys
# sys.path.append("/home/tomasgeffner/GeoDock/geodock/utils")
sys.path.append("/home/tomasgeffner/GeoDock")
from geodock.utils.esm_utils_struct import load_coords
from geodock.utils.pdb import save_PDB, place_fourth_atom 





class GeoDockDataset(data.Dataset):
    def __init__(
        self, 
        dataset: str = 'pinder',
        device: str = 'cuda',
    ):
        if dataset == 'pinder':
            self.data_dir = "/home/tomasgeffner/pinder_copy/splits_v2/"

            file_list = []
            for root, dirs, files in os.walk(self.data_dir):
                for f in files:
                    if ".pdb" in f:
                        file_list.append(os.path.join(root, f))
            self.file_list = file_list
        
        self.dataset = dataset
        self.device = device
        self.fail_list = []
        self.complexes_good_train = []
        self.complexes_good_test = []

        # Load esm
        # This to download: model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet('/home/tomasgeffner/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
        self.batch_converter = alphabet.get_batch_converter()
        self.esm_model = esm_model.to(device).eval()

    def __getitem__(self, idx: int):
        fail = False

        if self.dataset == 'pinder':
            pdb_file = self.file_list[idx]
            mode = pdb_file.split("/")[-2]
            full_complex = False
            if mode not in ["apo", "holo", "predicted", "alt"]:
                full_complex = True
            # coords, seq = load_coords(pdb_file, chain=None)
            try:
                # This line is sometimes problematic leads to some failures
                coords, seq, chain_lens = load_coords(pdb_file, chain=None)
                assert coords.shape[0] == sum(chain_lens), f"Chains and coords different lens, {coords.shape[0]}, {len(seq)} - {len(chain_lens)}, {sum(chain_lens)}\n{pdb_file}"
                if full_complex:
                    assert len(chain_lens) == 2, "Complex should have two chains"

                # Found non standard
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
            esm_rep = None

            # If single structure
            if not full_complex:
                data = HeteroData()
                
                esm_rep = self.get_esm_rep(seq)

                data['prot'].x = esm_rep
                data['prot'].pos = coords
                data['prot'].seq = seq
            
            else:
                data = HeteroData()

                split = chain_lens[0]

                data['receptor'].x = None
                data['receptor'].pos = coords[:split, :, :]
                data['receptor'].seq = seq[:split]

                data['ligand'].x = None
                data['ligand'].pos = coords[split:, :, :]
                data['ligand'].seq = seq[split:]


            data.name = pdb_file
            assert pdb_file[-4:] == ".pdb"
            out_name = pdb_file[:-4] + ".pt"
            torch.save(data, out_name)

            if full_complex:
                if "test" in pdb_file:
                    self.complexes_good_test.append(pdb_file.split("/")[-1][:-4])
                elif "train" in pdb_file:
                    self.complexes_good_train.append(pdb_file.split("/")[-1][:-4])

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

    # Generate the text files
    root = "/home/tomasgeffner/pinder_copy/"
    # need to check that for each correctly generated complex we have two holos, that's it
    root_train = "/home/tomasgeffner/pinder_copy/splits_v2/train/"
    clean_list = check_validity(complexes_good_train, root_train)



def check_validity(list_complexes, root):
    for pdb_file in list_complexes:
        print(pdb_file)


# "structure has multiple atoms with the same name" comes from
#   https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/inverse_folding/util.py#L105C35-L105C35
# For isntance, for this file /home/tomasgeffner/pinder_copy/splits/test/1jma__A1_Q92956--1jma__B1_P57083/apo/1jma__B1_P57083_L.pdb

# Update, this has to do with broken APOs, and a few others...

# Other error is "CSO" is a key error comes from ProteinSequence._dict_3to1[symbol.upper()] (I think symbol.upper is "CSO") comes from
#   https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/inverse_folding/util.py#L73C1-L73C1
# For isntance, for this file /home/tomasgeffner/pinder_copy/splits/test/2es4__A1_P0DUB8--2es4__B1_Q05490/2es4__A1_P0DUB8--2es4__B1_Q05490.pdb

# This has to do with non-standard aa
# Fixed by changing the function used to transform non standard to standard variants