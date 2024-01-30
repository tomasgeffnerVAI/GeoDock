"""Utility functions for working with pdbs

Adapted from Raptorx3DModelling/Common/PDBUtils.py
"""
import os
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import Bio
import numpy as np
import torch
from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from torch import Tensor
from Bio import SeqIO

import geodock.datasets.pinder_constants as pc


BLOSUM80 = substitution_matrices.load("BLOSUM80")
VALID_AA_3_LETTER_SET = set(pc.three_to_one_noncanonical_mapping.keys())


def default(x, y):
    return x if x is not None else y


def listToTuple(function):
    def wrapper(*args, **kwargs):
        args = [tuple(x) if isinstance(x, list) else x for x in args]
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        result = function(*args, **kwargs)
        # result = tuple(result) if type(result) == list else result
        return result

    return wrapper


def three_to_one(x):
    return (
        pc.three_to_one_noncanonical_mapping[x]
        if x in pc.three_to_one_noncanonical_mapping
        else "X"
    )


def load_fasta_file(seq_file, returnStr=True) -> Union[str, List]:
    """Load a fasta file.

    :param seq_file: file to read (fasta) sequence from.
    :param returnStr: whether to return string representation (default) or list.
    :return: sequence as string or list.
    """
    if not os.path.isfile(seq_file) or not seq_file.endswith(".fasta"):
        # pylint: disable-next=broad-exception-raised
        raise Exception("ERROR: an invalid sequence file: ", seq_file)
    record = SeqIO.read(seq_file, "fasta")
    return str(record.seq) if returnStr else record.seq


class SubMat:
    """Wrapper Around BLOSUM80 subst. matrix for handling
    non-standard residue types
    """

    @staticmethod
    def subst_matrix_get(aa1, aa2):
        """Heavy penalty for non-standard aas"""
        if (aa1, aa2) in BLOSUM80:
            val = BLOSUM80[(aa1, aa2)]
        elif (aa2, aa1) in BLOSUM80:
            val = BLOSUM80[(aa2, aa1)]
        else:
            val = -1000
        return val

    def __contains__(self, *item):
        return True

    def __getitem__(self, item):
        return self.subst_matrix_get(*item)


class PDBExtractException(Exception):
    pass


def get_structure_parser(pdb_file: str) -> Union[PDBParser, MMCIFParser]:
    """gets a parser for the underlying pdb structure

    :param pdb_file: the file to obtain a structure parser for
    :return: structure parser for pdb input
    """
    is_pdb, is_cif = [pdb_file.endswith(x) for x in (".pdb", ".cif")]
    assert (
        is_pdb or is_cif
    ), f"ERROR: pdb file must have .cif or .pdb type, got {pdb_file}"
    return MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)


@lru_cache(maxsize=16)
def get_structure(pdbfile: str, name: str = None):
    """Get BIO.Structure object"""
    parser = get_structure_parser(pdbfile)
    name = default(name, os.path.basename(pdbfile))
    return parser.get_structure(name, pdbfile)


def extract_pdb_seq_from_residues(
    residues: List[Residue],
) -> Tuple[str, List[Residue]]:
    """
    extract a list of residues with valid 3D coordinates excluding
    non-standard amino acids
    returns the amino acid sequence as well as a list of residues
    with standard amino acids
    """
    valid_list = VALID_AA_3_LETTER_SET

    def fltr(r):
        return is_aa(r, standard=True) and r.get_resname().upper() in valid_list  # noqa

    residueList = list(filter(fltr, residues))
    res_names = list(map(lambda x: x.get_resname(), residueList))
    pdbseq = "".join(list(map(three_to_one, res_names)))
    return pdbseq, residueList


def extract_pdb_seq_by_chain(
    structure: Structure, chain_id: Optional[str] = None
) -> Tuple[List, ...]:
    """extract sequences and residue lists for each chain
    :return: pdbseqs, residue lists and also the chain objects
    """
    model = structure[0]
    pdbseqs, residueLists, chains = [], [], []
    for chain in model:
        if chain.get_id() != default(chain_id, chain.get_id()):
            continue
        residues = list(chain.get_residues())
        pdbseq, residueList = extract_pdb_seq_from_residues(residues)
        pdbseqs.append(pdbseq)
        residueLists.append(residueList)
        chains.append(chain)
    return pdbseqs, residueLists, chains


@lru_cache(maxsize=8)
def extract_pdb_seq_from_pdb_file(
    pdbfile: str,
    name: Optional[str] = None,
    chain_id: Optional[str] = None,
) -> Tuple[List, ...]:
    """Extract sequences and residue lists from pdbfile for all the chains
    :param pdbfile: pdb file to extract from
    :param name: name for bio.pdb structure
    :return: lists of : pdbseqs, residueLists, chains from each
    chain in input pdb file
    """
    name = default(name, os.path.basename(pdbfile)[:-4])
    structure = get_structure(pdbfile=pdbfile, name=name)
    return extract_pdb_seq_by_chain(structure, chain_id=chain_id)


def extract_seq_from_pdb_n_chain_id(
    pdbfile: str, chain_id: str, name: str = None
) -> str:
    """Extract the sequence of a specific pdb chain"""
    name = default(name, os.path.basename(pdbfile)[:-4])
    structure = get_structure(pdbfile=pdbfile, name=name)
    model = structure[0]
    chain_ids = []
    for chain in model:
        chain_ids.append(chain.get_id())
        residues = chain.get_residues()
        if chain.get_id() == chain_id:
            pdbseq, _ = extract_pdb_seq_from_residues(residues)
            return pdbseq
    raise PDBExtractException(
        f"No chain with id {chain_id}, found chains: {[chain_ids]}"
    )


def calc_num_mismatches(alignment) -> Tuple[int, int]:
    """Calculate number of mismatches in sequence alignment

    :param alignment: sequence alignment(s)
    :return: number of mismatches in sequence alignment
    """
    S1, S2 = alignment[:2]
    numMisMatches = np.sum(
        [
            a != b
            for a, b in zip(S1, S2)
            if a != "-" and b != "-" and a != "X" and b != "X"
        ]
    )
    numMatches = np.sum([a == b for a, b in zip(S1, S2) if a != "-" and a != "X"])
    return int(numMisMatches), int(numMatches)


def alignment_to_mapping(alignment) -> List:
    """Convert sequence alignment to residue-wise mapping

    :param alignment: sequence alignment
    :return: mapping
    """
    S1, S2 = alignment[:2]
    # convert an aligned seq to a binary vector with 1 indicates
    # aligned and 0 gap.
    y = np.array([1 if a != "-" else 0 for a in S2])
    # get the position of each residue in the original sequence,
    # starting from 0.
    ycs = np.cumsum(y) - 1
    np.putmask(ycs, y == 0, -1)
    # map from the 1st seq to the 2nd one. set -1 for an unaligned residue
    # in the 1st sequence.
    mapping = [y0 for a, y0 in zip(S1, ycs) if a != "-"]
    return mapping


def map_seq_to_residue_list(
    sequence: str, pdbseq: str, residueList: List[Residue]
) -> Tuple[Optional[List], Optional[int], Optional[int]]:
    """map one query sequence to a list of PDB residues by sequence alignment
    pdbseq and residueList are generated by ExtractPDBSeq or
    ExtractPDBSeqByChain from a PDB file
    :param sequence:
    :param pdbseq:
    :param residueList:
    :return: seq2pdb mapping, numMisMatches and numMatches
    """
    # here we align PDB residues to query sequence instead
    # of query to PDB residues
    if sequence != pdbseq:
        alignments = pairwise2.align.localds(
            pdbseq,
            sequence,
            BLOSUM80,
            -5,
            -0.2,
        )
    else:
        # pairwise2.align.localds takes ~0.5 seconds for a 300 AA sequence
        # this saves a lot of time in data loading
        alignments = [Bio.pairwise2.Alignment(pdbseq, pdbseq, 100, 0, len(pdbseq))]
    if not bool(alignments):
        return None, None, None
    # find the alignment with the minimum difference
    diffs = []
    for alignment in alignments:
        mapping_pdb2seq, diff = alignment_to_mapping(alignment), 0
        for current_map, prev_map, current_residue, prev_residue in zip(
            mapping_pdb2seq[1:],
            mapping_pdb2seq[:-1],
            residueList[1:],
            residueList[:-1],
        ):
            # in principle, every PDB residue with valid 3D coordinates
            # shall appear in the query sequence. otherwise, apply a big penalty
            if current_map < 0:
                diff += 10
                continue

            if prev_map < 0:
                continue

            # calculate the difference of sequence separation in both
            # the PDB seq and the query seq. the smaller, the better
            current_id = current_residue.get_id()[1]
            prev_id = prev_residue.get_id()[1]
            id_diff = max(1, current_id - prev_id)
            map_diff = current_map - prev_map
            diff += abs(id_diff - map_diff)

        numMisMatches, numMatches = calc_num_mismatches(alignment)
        diffs.append(diff - numMatches)

    diffs = np.array(diffs)
    alignment = alignments[diffs.argmin()]

    numMisMatches, numMatches = calc_num_mismatches(alignment)

    # map from the query seq to pdb
    mapping_seq2pdb = alignment_to_mapping((alignment[1], alignment[0]))

    return mapping_seq2pdb, numMisMatches, numMatches


def map_seq_to_pdb_chain(
    sequence,
    pdbfile,
    chain_id: Optional[str] = None,
):
    """Maps sequence to a pdb file, selecting the sequence from
    chain with most matching aligned residues.
    Parameters:
        sequence: sequence (string)
        pdbfile: pdb file to map to
        chain_is (optional): The pdb chain id to map to.

    :return: seq2pdb mapping, the pdb residue list, the pdb seq, the pdb chain,
    the number of mismtaches and matches
    """
    if not os.path.isfile(pdbfile):
        raise PDBExtractException(
            "ERROR: the pdb file does not exist:\n\t", pdbfile, "\nType:\n\t", pdbfile
        )

    # extract PDB sequences by chains
    pdbseqs, residue_lists, chains = extract_pdb_seq_from_pdb_file(
        pdbfile, chain_id=chain_id
    )

    best_pdb_seq = None
    best_mapping = None
    best_residue_list = None
    best_chain = None
    min_mismatches = np.iinfo(np.int32).max
    max_matches = np.iinfo(np.int32).min
    for pdbseq, residue_list, chain in zip(pdbseqs, residue_lists, chains):
        if chain.get_id() != default(chain_id, chain.get_id()):
            # skip over chains without specified id
            continue

        seq2pdb_mapping, num_mismatches, num_matches = map_seq_to_residue_list(
            sequence, pdbseq, residue_list
        )
        if seq2pdb_mapping is None:
            continue
        if max_matches < num_matches:
            # if numMisMatches < minMisMatches:
            best_mapping = seq2pdb_mapping
            min_mismatches = num_mismatches
            max_matches = num_matches
            best_residue_list = residue_list
            best_pdb_seq = pdbseq
            best_chain = chain

    return (
        best_mapping,
        best_residue_list,
        best_pdb_seq,
        best_chain,
        min_mismatches,
        max_matches,
    )


def extract_coords_by_mapping(seq2pdb_mapping, residueList, atom_tys):
    """Extract coordinates from residue list by sequence mapping"""
    needed_atoms = [a.upper() for a in atom_tys]
    needed_atom_set = set(needed_atoms)
    atom_coordinates = []
    for aligned_pos in seq2pdb_mapping:
        coordinates = defaultdict(lambda: None)

        if aligned_pos >= 0:
            res = residueList[aligned_pos]
            for atom in res:
                atom_name = atom.get_id().upper()
                if atom_name in needed_atom_set:
                    coordinates[atom_name] = atom.get_vector()
            atom_coordinates.append(coordinates)
        else:
            atom_coordinates.append(None)

    return atom_coordinates


def extract_chain_coordinates_from_seq_n_pdb(
    sequence: str,
    pdbfile: str,
    atom_tys: Union[Tuple[str], List[str]],
    chain_id: Optional[str] = None,
):
    """
    :param sequence: sequence to map from
    :param pdbfile: pdb file to extract from
    :param atom_tys: atom types to extract coords for.
    :param maxMisMatches: maximum number of allowed mismatches
      in seq to pdb_seq alignment
    :param minMatchRatio: minimum allowed match ratio
    :return: tuple containining:
        (1) atom_coordinates : atom coordinates for each residue
          in the input sequence
        (2) pdb_seq : pdb sequence for chain which was mapped to
        (3) num_mismatches: number of mismatches in alignment
        (4) num_matches: number of matches in alignment
        (5) residue ids from mapped pdb file
    """
    out = map_seq_to_pdb_chain(
        sequence=sequence,
        pdbfile=pdbfile,
        chain_id=chain_id,
    )
    (
        seq2pdb_mapping,
        residue_list,
        pdbseq,
        _,  # chain
        num_mismatches,
        num_matches,
    ) = out
    if seq2pdb_mapping is None:
        return None, None, None, None, None
    residue_list = list(residue_list)
    atom_coordinates = extract_coords_by_mapping(
        seq2pdb_mapping=seq2pdb_mapping,
        residueList=residue_list,
        atom_tys=atom_tys,
    )
    res_ids = [r.get_id()[1] for r in residue_list]
    return (
        atom_coordinates,
        pdbseq,
        num_mismatches,
        num_matches,
        res_ids,
    )


@listToTuple
@lru_cache(8)
def extract_pdb_data_by_seq_mapping(
    seq: Optional[str],
    pdb_path: str,
    atom_tys: List[str],
    chain_id: Optional[str] = None,
) -> Tuple[Tensor, Tensor, List, int]:
    """Extracts data from a pdb file by mapping and aligning to the
    given sequence.

    Parameters:
        seq: protein sequence of length n to map from
        pdb_path: pdb path to extract coordinates from
        atom_tys: atom types to extract coordinates for

    Returns:
    Tuple containing
        (1) atom coords: Tensor of shape (n,a,3) containing atom coordinates for
        each atom type (1..a) and each residue 1..n in the input sequence
        (2) atom mask: Tensor of shape (n,a) indicating whether valid coordinates
        were obtained for atom types for each atom in atom_tys (1..a) and each
        residue (1..n) in the input sequence
        (3) List of residue ids extracted from the pdb file, (length n, with
        "None" inserted at unaligned positions)
        (4) Number of mismatched residues in the alignment between
        the query sequence and the pdb sequence.
    """
    assert seq is not None, "must provide sequence for alignment of pdb file!"
    out = extract_chain_coordinates_from_seq_n_pdb(
        sequence=seq,
        pdbfile=pdb_path,
        atom_tys=atom_tys,
        chain_id=chain_id,
    )

    atom_coords, _, num_mismatches, _, res_ids = out
    atom_coords, atom_mask, res_ids = _get_coord_ids_n_masks(
        atom_coords, res_ids, atom_tys
    )
    return atom_coords, atom_mask, res_ids, num_mismatches


def _get_coord_ids_n_masks(
    atom_coords: List[Dict[str, np.ndarray]], res_ids: List[str], atom_tys: List[str]
) -> Tuple[Tensor, Tensor, List]:
    n_res, n_atoms = len(atom_coords), len(atom_tys)
    coords, mask, ids = (
        torch.zeros(n_res, n_atoms, 3),
        torch.zeros(n_res, n_atoms),
        [None] * n_res,
    )
    pdb_idx = 0
    for seq_idx, res in enumerate(atom_coords):
        if res is None:
            continue
        for atom_pos, atom_ty in enumerate(atom_tys):
            if res[atom_ty] is None:
                continue
            coords[seq_idx, atom_pos] = torch.tensor(
                [res[atom_ty][j] for j in range(3)]
            )
            mask[seq_idx, atom_pos] = 1
        ids[seq_idx] = res_ids[pdb_idx]
        pdb_idx += 1
    return coords, mask.bool(), ids