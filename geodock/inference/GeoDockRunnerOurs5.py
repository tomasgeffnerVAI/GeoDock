# Stuff to change when aligned / not aligned:
# (1) output folds, (2) cuda device, (3) decoy stuff in get example


import sys
sys.path.append("/home/celine/GeoDock")
import os
# import esm
import torch
from time import time
# from geodock.utils.embed import embed
from geodock.utils.embed import get_pair_mats, get_pair_relpos
from geodock.utils.docking import dock
from geodock.model.GeoDock import GeoDock
# from esm.inverse_folding.util import load_coords
from geodock.datasets.pinder_dataset_utils import get_example_from_pdbs_n_sequence, accept_example
from geodock.model.interface import GeoDockInput
import geodock.datasets.protein_constants as pc
import pandas as pd


def get_example(
    decoy_receptor_pdb,
    decoy_ligand_pdb,
    target_pdb,
    batch_converter,
    device,
):
    data = get_example_from_pdbs_n_sequence(
        seq_paths=[None, None],
        decoy_pdb_paths=[decoy_receptor_pdb, decoy_ligand_pdb],
        target_pdb_paths=[target_pdb, target_pdb],
        # TODO: make atom types in same order as geodock!
        # atom_tys=tuple(pc.ALL_ATOMS),
        atom_tys=tuple(pc.BB_ATOMS_GEO),
        decoy_chain_ids=["R", "L"],
        target_chain_ids=["R", "L"],
    )

    # NOTE: the follow must be commented out or apo/predicted might not work

    # if not accept_example(data):
    #     print("not accepted!")
    #     return None, None
    
    chain1_mask = (
        data["target"]["residue_mask"][0] & data["decoy"]["residue_mask"][0]
    )
    chain2_mask = (
        data["target"]["residue_mask"][1] & data["decoy"]["residue_mask"][1]
    )
    coords1_true = data["target"]["coordinates"][0][chain1_mask]
    coords2_true = data["target"]["coordinates"][1][chain2_mask]
    coords1_decoy = data["decoy"]["coordinates"][0][chain1_mask]
    coords2_decoy = data["decoy"]["coordinates"][1][chain2_mask]
    
    seq1 = "".join(
        [x for x, m in zip(data["target"]["sequence"][0], chain1_mask) if m]
    )
    seq2 = "".join(
        [x for x, m in zip(data["target"]["sequence"][1], chain2_mask) if m]
    )
    decoy_coords = torch.cat([coords1_decoy, coords2_decoy], dim=0)
    input_pairs = get_pair_mats(decoy_coords, len(seq1))
    input_contact = torch.zeros(*input_pairs.shape[:-1])[..., None]
    pair_embeddings = torch.cat([input_pairs, input_contact], dim=-1).to(device)
    positional_embeddings = get_pair_relpos(len(seq1), len(seq2)).to(device)
    *_, tokens = batch_converter([("1", seq1), ("2", seq2)])
    gd_input =  GeoDockInput(
        pair_embeddings=pair_embeddings.unsqueeze(0),
        positional_embeddings=positional_embeddings.unsqueeze(0),
        seq1=[seq1],
        seq2=[seq2],
        esm_tokens=tokens.unsqueeze(0),
    )
    true_coords = (coords1_true, coords2_true)

    # NOTE: the following must be commented out because .-.
    
    # if len(seq1) > 1200 or len(seq2) > 1200:
    #     print("too long!")
    #     return None, None
    
    return gd_input, true_coords

class GeoDockRunner():
    """
    Wrapper for GeoDock model predictions.
    """
    def __init__(self, ckpt_file, run_mode):

        # Check if gpu is available
        if run_mode == "GPU":
            self.device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        elif run_mode == "CPU":
            self.device = torch.device("cpu")
        print(f"Running on: {self.device}")

        # Load GeoDock model
        self.model = GeoDock.load_from_checkpoint(ckpt_file, map_location=self.device).eval().to(self.device)
        _, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = alphabet.get_batch_converter()

    def dock(
        self,
        complex_id,
        mode_r,
        decoy_receptor_pdb,
        decoy_ligand_pdb,
        target_pdb,
        out_name,
        do_refine=True,
        use_openmm=True,
    ):
        gd_input, true_coords = get_example(
            decoy_receptor_pdb,
            decoy_ligand_pdb,
            target_pdb,
            self.batch_converter,
            device=self.device,
        )
        if gd_input is None:
            return
        # Start docking
        dock(
            complex_id,
            mode_r,
            out_name,
            gd_input.seq1[0],
            gd_input.seq2[0],
            gd_input,
            self.model,
            do_refine=do_refine,
            use_openmm=use_openmm,
            true_coords=true_coords,
        )

if __name__ == '__main__':

    # Define modes
    testset = "pinder_xl"
    refinement_mode = False
    run_mode = "CPU" #CPU
    structure_modes = ["apo", "holo", "predicted"] #["holo", "apo", "predicted"]

    # Load paths
    ckpt_file = "/home/celine/GeoDock/logs/runs/2023-11-29/20-49-55/checkpoints/last.ckpt"  # Add checkpoint here
    root = "/mnt/disks/pinder-us-east5-a-2024-01-09/pdbs"
    # Load index file
    df = pd.read_csv("/mnt/disks/pinder-us-east5-a-2024-01-09/index.csv.gz", compression='gzip')
    
    print("======\nDocking\n======\n")
    print(f"Set: {testset}")
    print(f"Refinement: {refinement_mode}" )
    print(f"Run mode: {run_mode}")

    # Generate lists of target/decoy paths and respective structure modes from index file
    pdb_paths = []
    modes_decoy = []

    for mode in structure_modes:

        # load only certain test set with subgroup apo/holo/predicted
        filtered_df = df[(df[testset]) & (df[f'{mode}_R']) & (df[f'{mode}_L']) ] #| (df['pinder_af2'] == True)]
        # load id of full complex (target)
        id_values = filtered_df['id'].tolist() #['7zvj__A1_O95461--7zvj__B1_O95461']

        # load ids of R/L
        id_values_R = filtered_df[f'{mode}_R_pdb'].tolist()
        id_values_L = filtered_df[f'{mode}_L_pdb'].tolist()

        # list of complex paths (targets)
        dirs_complexes = [f + ".pdb" for f in id_values if os.path.isdir(os.path.join(root, f))]
        # list of R/L paths
        dirs_receptors = [f for f in id_values_R if os.path.isdir(os.path.join(root, f))]
        dirs_ligands = [f for f in id_values_L if os.path.isdir(os.path.join(root, f))]

        # iterate over all complexes (targets):
        for index, (id_value, id_R, id_L) in enumerate(zip(id_values, id_values_R, id_values_L)):

            path_complex = os.path.join(os.path.join(root, id_value + ".pdb"))
            path_receptor = os.path.join(os.path.join(root, id_R))
            path_ligand = os.path.join(os.path.join(root, id_L))

            # check if files exists:
            if not os.path.isfile(path_complex):
                print(f"The file {path_complex} does not exists.")
                continue
            elif not os.path.isfile(path_receptor):
                print(f"The file {path_receptor} does not exists.")
                continue
            elif not os.path.isfile(path_ligand):
                print(f"The file {path_ligand} does not exists.")
                continue
            
            pdb_paths.append((id_value, path_complex, path_receptor, path_ligand))                
            modes_decoy.append((mode, mode))  # For now same mode for both
        print(f"{mode} loaded. Processed {len(id_values)} complexes.")

    # Load model
    geodock = GeoDockRunner(ckpt_file=ckpt_file, run_mode=run_mode)

    # iterate over paths list
    count = 0
    for files, dmodes in zip(pdb_paths, modes_decoy):
        count += 1
        
        complex_name, complex_pdb, receptor_pdb, ligand_pdb = files

        # Check if files exists
        if not os.path.exists(receptor_pdb):
            print(f'{receptor_pdb} does not exist')
        if not os.path.exists(ligand_pdb):
            print(f'{ligand_pdb} does not exist')
            
        mode_r, mode_l = dmodes
        out_name = f"{complex_name}/{mode_r}_decoys/model_1"

        print(count, "=======")
        print(mode_r, complex_name)
        print(receptor_pdb, ligand_pdb)
        
        try:
            pred = geodock.dock(
                complex_id = complex_name,
                mode_r = mode_r,
                decoy_receptor_pdb=receptor_pdb,
                decoy_ligand_pdb=ligand_pdb,
                target_pdb=complex_pdb,
                out_name=out_name,
                do_refine=refinement_mode,  # Original is true, but Matt said we dont need
                use_openmm=refinement_mode,  # Original is true, but Matt said we dont need
            )
        except Exception as e:
            print(e)
