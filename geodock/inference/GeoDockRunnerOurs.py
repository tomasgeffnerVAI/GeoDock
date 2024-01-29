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
        # target_pdb_paths=[decoy_receptor_pdb, decoy_ligand_pdb],
        target_pdb_paths=[target_pdb, target_pdb],
        # TODO: make atom types in same order as geodock!
        # atom_tys=tuple(pc.ALL_ATOMS),
        atom_tys=tuple(pc.BB_ATOMS_GEO),
        decoy_chain_ids=["R", "L"],
        target_chain_ids=["R", "L"],
    )
    if not accept_example(data):
        return None, None
    
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
    
    if len(seq1) > 1200 or len(seq2) > 1200:
        return None, None
    
    return gd_input, true_coords

class GeoDockRunner():
    """
    Wrapper for GeoDock model predictions.
    """
    def __init__(self, ckpt_file):
        # Check if gpu is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load GeoDock model
        self.model = GeoDock.load_from_checkpoint(ckpt_file, map_location=self.device).eval().to(self.device)
        _, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = alphabet.get_batch_converter()
    
    def dock(
        self,
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
    ckpt_file = "/home/celine/logs/runs/2023-10-26/15-56-36/checkpoints/last.ckpt"  # Add checkpoint here
    # root = "/home/celine/pinder-public/splits/test"
    root = "/home/celine/data_run_inference/test"
    # dirs_complexes = [f.path for f in os.scandir(root) if f.is_dir()]
    dirs_complexes = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    pdb_paths = []
    modes_decoy = []
    for d in dirs_complexes:
        print(f"Adding {d}")
        root_complex = os.path.join(root, d)
        complex_pdb = os.path.join(root_complex, d + ".pdb")
        if not os.path.isfile(complex_pdb):
            print(f"{complex_pdb} not there")
            continue
        modes = ["apo", "holo", "predicted"]  # Ignore alt
        for mode in modes:
            root_decoys = os.path.join(root_complex, mode)
            if not os.path.isdir(root_decoys):
                print(f"Mode {mode} not available")
                continue
            receptor_decoy_pdb = [os.path.join(root_decoys, f) for f in os.listdir(root_decoys) if f.endswith("R.pdb")]
            ligand_decoy_pdb = [os.path.join(root_decoys, f) for f in os.listdir(root_decoys) if f.endswith("L.pdb")]
            if len(receptor_decoy_pdb) == 0 or len(ligand_decoy_pdb) == 0:
                print(f"Mode {mode} has {len(receptor_decoy_pdb)} receptors and {len(ligand_decoy_pdb)} ligands")
                continue
            receptor_decoy_pdb = receptor_decoy_pdb[0]
            ligand_decoy_pdb = ligand_decoy_pdb[0]
            pdb_paths.append((d, complex_pdb, receptor_decoy_pdb, ligand_decoy_pdb))
            modes_decoy.append((mode, mode))  # For now same mode for both
    
    print("======\nDocking\n======\n")
    
    geodock = GeoDockRunner(ckpt_file=ckpt_file)
    count = 0
    for files, dmodes in zip(pdb_paths, modes_decoy):
        count += 1
        # if count >= 5:
        #     break
        
        complex_name, complex_pdb, receptor_pdb, ligand_pdb = files
        mode_r, mode_l = dmodes
        out_name = f"{complex_name}_{mode_r}_{mode_l}"
        print(count, "=======")
        
        try:
            pred = geodock.dock(
                decoy_receptor_pdb=receptor_pdb,
                decoy_ligand_pdb=ligand_pdb,
                target_pdb=complex_pdb,
                out_name=out_name,
                do_refine=False,  # This? Should we refine? They had it to true
                use_openmm=False,  # True
            )
        except Exception as e:
            print(e)
