"""Protein Complex dataset"""
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from einops import repeat
import torch

import geodock.datasets.pinder_constants as pc
from geodock.datasets.pinder_pdb_utils import (
    extract_pdb_data_by_seq_mapping,
    calc_num_mismatches,
    BLOSUM80,
    pairwise2,
    load_fasta_file,
    extract_pdb_seq_from_pdb_file,
)
import logging


log = logging.getLogger(__name__)


def exists(x):
    return x is not None


def default(x, y):
    return x if exists(x) else y


def cast_defaultdict(d):
    if isinstance(d, dict) or isinstance(d, defaultdict):
        return {k: cast_defaultdict(v) for k, v in d.items()}
    return d


@lru_cache(16)
def safe_load_sequence(
    seq_path: Optional[str], pdb_path: str, chain_id: Optional[str] = None
) -> str:
    """Loads sequence, either from fasta or given pdb file"""
    if exists(seq_path):
        pdbseqs = [load_fasta_file(seq_path)]
    else:
        pdbseqs, *_ = extract_pdb_seq_from_pdb_file(pdb_path, chain_id=chain_id)
    if len(pdbseqs) > 1 and not exists(chain_id):
        log.warn(f"Multiple chains found for pdb: {pdb_path}")
    return pdbseqs[0]


def get_example_from_pdbs_n_sequence(
    seq_paths: List[Optional[str]],
    decoy_pdb_paths: List[str],
    target_pdb_paths: List[Optional[str]],
    atom_tys: Optional[List[str]] = None,
    decoy_chain_ids: Optional[List[str]] = None,
    target_chain_ids: Optional[List[str]] = None,
    align_seq_to: str = "target",  # target or decoy
) -> Dict:
    """Load data given native and decoy pdb paths and sequence path

    Output is a dictionary with keys "target" and "decoy"
    Typically, the data within the "decoy" sub-dict is used
    to generate features/train the model, and the "target"
    sub dict is used as ground-truth labels.

    For example, this format allows the user to do things like
    train on unbound chain conformations (decoy data) and predict bound
    conformations (native data).
    """
    atom_tys = tuple(default(atom_tys, pc.ALL_ATOMS))
    decoy_chain_ids = default(decoy_chain_ids, [None] * len(decoy_pdb_paths))
    target_chain_ids = default(target_chain_ids, [None] * len(decoy_pdb_paths))

    batch = dict(
        metadata=dict(
            atom_tys=list(atom_tys),
            decoy_pdb_paths=decoy_pdb_paths,
            target_pdb_paths=target_pdb_paths,
            seq_paths=seq_paths,
        )
    )

    target_data, decoy_data = defaultdict(list), defaultdict(list)

    for seq_path, decoy_pdb, tgt_pdb, decoy_cid, tgt_cid in zip(
        seq_paths,
        decoy_pdb_paths,
        target_pdb_paths,
        decoy_chain_ids,
        target_chain_ids,
    ):
        aln_kwargs = dict(
            pdb_path=decoy_pdb if align_seq_to == "decoy" else tgt_pdb,
            chain_id=decoy_cid if align_seq_to == "decoy" else tgt_cid,
        )
        seq = safe_load_sequence(seq_path, **aln_kwargs)
        # print("Seq path + seq", seq_path, seq)

        decoy_data = append_chain_to_data(
            decoy_data,
            pdb_path=decoy_pdb,
            seq=seq,
            atom_tys=atom_tys,
            chain_id=decoy_cid,
        )

        target_data = append_chain_to_data(
            target_data,
            pdb_path=default(tgt_pdb, decoy_pdb),
            seq=seq,
            atom_tys=atom_tys,
            chain_id=tgt_cid,
        )
        if default(tgt_pdb, decoy_pdb) == decoy_pdb:
            target_data = deepcopy(target_data)

    # create alternate alignment for homodimers
    alt_target_data = deepcopy(target_data)
    batch["homodimer"] = False
    # If protein complex
    if len(decoy_pdb_paths) == 2:
        batch["homodimer"] = False
        # swap and re-align chains
        if is_homodimer(*decoy_data["sequence"]):
            batch["homodimer"] = True
            alt_target_data = defaultdict(list)
            for seq, tgt_pdb, tgt_cid in zip(
                decoy_data["sequence"],
                reversed(target_pdb_paths),
                reversed(target_chain_ids),
            ):
                alt_target_data = append_chain_to_data(
                    alt_target_data,
                    pdb_path=tgt_pdb,
                    seq=seq,
                    atom_tys=atom_tys,
                    chain_id=tgt_cid,
                )

    batch["decoy"], batch["target"], batch["alt_target"] = map(
        cast_defaultdict, (decoy_data, target_data, alt_target_data)
    )
    batch["decoy"]["pdb_paths"] = decoy_pdb_paths
    batch["target"]["pdb_paths"] = target_pdb_paths
    batch["alt_target"]["pdb_paths"] = list(reversed(target_pdb_paths))
    return batch


def append_chain_to_data(
    data_dict: Dict,
    pdb_path: str,
    seq: str,
    atom_tys: Tuple[str],
    chain_id: Optional[str] = None,
) -> Dict:
    # target coords and mask
    crds, mask, res_ids, mismatches = extract_pdb_data_by_seq_mapping(
        seq, pdb_path=pdb_path, atom_tys=atom_tys, chain_id=chain_id
    )
    seq_encoding = torch.tensor([pc.AA_TO_INDEX[x] for x in seq])
    canonical_mask = aa_to_canonical_atom_mask(atom_tys)[seq_encoding]
    atom_mask = mask & canonical_mask
    ca_crds = repeat(
        crds[:, min(1, len(atom_tys)), :],
        "i c -> i a c",
        a=len(atom_tys),
    )
    crds[~atom_mask] = ca_crds[~atom_mask]

    data_dict["sequence"].append(seq)
    data_dict["res_ids"].append(res_ids)

    data_dict["tokenized_sequence"].append(seq_encoding.long())
    data_dict["atom_mask"].append(atom_mask)

    data_dict["coordinates"].append(crds)
    data_dict["sequence_mismatches"].append(mismatches)
    data_dict["residue_mask"].append(
        torch.all(mask[:, backbone_atom_tensor(atom_tys)], dim=-1)
    )
    return data_dict


@lru_cache(maxsize=1)
def aa_to_canonical_atom_mask(atom_tys: Tuple[str]):
    msk = torch.zeros(len(pc.INDEX_TO_AA_THREE), len(atom_tys))
    for aa_idx in range(len(pc.INDEX_TO_AA_THREE)):
        aa = pc.INDEX_TO_AA_THREE[aa_idx]
        for atom_idx, atom in enumerate(atom_tys):
            if atom in pc.BB_ATOMS:
                msk[aa_idx, atom_idx] = 1
            elif atom in pc.AA_TO_SC_ATOMS[aa]:
                msk[aa_idx, atom_idx] = 1
    return msk.bool()


@lru_cache(maxsize=1)
def backbone_atom_tensor(atom_tys: Tuple[str]):
    valid_bb_atoms = set(atom_tys).intersection(set(pc.BB_ATOMS))
    return torch.tensor([atom_tys.index(x) for x in valid_bb_atoms]).long()


def is_homodimer(
    chain_1_seq: str,
    chain_2_seq: str,
    min_seq_id: float = 0.9,
) -> bool:
    homo = True
    min_len = min(len(chain_1_seq), len(chain_2_seq))
    max_len = max(len(chain_1_seq), len(chain_2_seq))
    if (min_len / max_len) < min_seq_id:
        homo = False
    if homo:
        aln = pairwise2.align.localds(chain_1_seq, chain_2_seq, BLOSUM80, -5, -0.2)
        matches = 0
        if len(aln) > 0:
            _, matches = calc_num_mismatches(aln[0])
        homo = (matches / min_len) > min_seq_id
    return homo


def accept_example(
    example: Dict,
    min_chain_len: float = 30.0,
    min_backbone_percentage: float = 90.0,
    max_mask_percentage: float = 10.0,
    max_mismatch_percentage: float = 10.0,
    max_mismatches: float = 10.0,
    min_ca_contacts: Optional[float] = 10.0,
    contact_threshold_angstroms: float = 10.0,
) -> bool:
    # aligned target coordinate data that has mapping in both decoy and target chains
    common_aligned_coords = []
    num_chains = len(example["target"]["coordinates"])

    # establish valid BB atoms -- CA is always assumed to be present
    atom_tys = example["metadata"]["atom_tys"]
    ca_posn = example["metadata"]["atom_tys"].index("CA")
    bb_atom_posns = backbone_atom_tensor(tuple(atom_tys))

    # iterate over all chains and apply chain-level filters
    for idx in range(len(example["decoy"]["sequence"])):
        # determine common residues between decoy and target chains
        decoy_residue_mask = example["decoy"]["residue_mask"][idx]
        target_residue_mask = example["target"]["residue_mask"][idx]
        residue_mask = decoy_residue_mask & target_residue_mask
        total_common_residues = torch.sum(residue_mask)

        # determine common atoms
        decoy_atom_mask = example["decoy"]["atom_mask"][
            idx
        ] & decoy_residue_mask.unsqueeze(-1)
        target_atom_mask = example["target"]["atom_mask"][
            idx
        ] & target_residue_mask.unsqueeze(-1)
        atom_mask = target_atom_mask & decoy_atom_mask

        # check chain-level filters

        # min chain length (if no common residues, always discard)
        if total_common_residues < max(1, min_chain_len):
            return False

        # backbone atom coverage
        num_res_with_all_bb_atoms = torch.sum(
            torch.all(atom_mask[:, bb_atom_posns], dim=-1)
        )
        if (num_res_with_all_bb_atoms / total_common_residues) < (
            min_backbone_percentage / 100
        ):
            return False

        # fraction of masked residues
        if (1 - total_common_residues / residue_mask.shape[0]) > (
            max_mask_percentage / 100
        ):
            return False

        # sequence mismatches
        n_mismatch = max(
            example["decoy"]["sequence_mismatches"][idx],
            example["target"]["sequence_mismatches"][idx],
        )
        mismatch_frac = n_mismatch / total_common_residues

        if mismatch_frac > (max_mismatch_percentage / 100):
            return False

        if n_mismatch > max_mismatches:
            return False

        # store the coordinates for computing contacts
        ca_mask = atom_mask[:, ca_posn]
        common_aligned_coords.append(
            example["target"]["coordinates"][idx][ca_mask][:, ca_posn]
        )

    if min_ca_contacts is None or num_chains == 1:
        return True
    # if there are multiple chains, then filter contacts
    assert num_chains == 2, "Contact filtering only applied to Dimers!"
    # contacts
    # using cdist with matmul is extremely unstable
    dists = torch.cdist(
        *common_aligned_coords, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    )
    num_contacts = torch.sum(dists < contact_threshold_angstroms)
    return num_contacts >= min_ca_contacts