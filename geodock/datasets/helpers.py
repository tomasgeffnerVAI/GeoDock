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
import geodock.datasets.protein_constants as pc
from Bio import SeqIO
from einops import rearrange, repeat
from collections import defaultdict
import copy


AA3LetterCode = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "ASX",
    "CYS",
    "GLU",
    "GLN",
    "GLX",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
]
AA1LetterCode = [
    "A",
    "R",
    "N",
    "D",
    "B",
    "C",
    "E",
    "Q",
    "Z",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
    "-",
]

BLOSUM80 = substitution_matrices.load("BLOSUM80")
VALID_AA_3_LETTER_SET = set(AA3LetterCode)
THREE_TO_ONE = {three: one for three, one in zip(AA3LetterCode, AA1LetterCode)}
ONE_TO_THREE = {one: three for three, one in THREE_TO_ONE.items()}


def exists(x):
    return x is not None


def default(x, y):
    return x if exists(x) else y


def three_to_one(x):
    return THREE_TO_ONE[x] if x in THREE_TO_ONE else "X"


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
    #print("hello", model)
    print([chain.get_id() for chain in chains])
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
    # print(pdbfile, chain_id)
    name = default(name, os.path.basename(pdbfile)[:-4])
    structure = get_structure(pdbfile=pdbfile, name=name)
    #print(len(structure))
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


def map_seq_to_pdb(
    sequence,
    pdbfile,
    chain_id: Optional[str] = None,
    maxMisMatches=None,
    minMatchRatio=0.5,
):
    """Maps sequence to a pdb file,
      selecting the sequence from chain with best match.
    :param sequence: sequence (string)
    :param pdbfile: pdb file to map to
    :param maxMisMatches: max allowed number of mismatches
    :param minMatchRatio: the minimum ratio of matches on the query sequence
    :return: seq2pdb mapping, the pdb residue list, the pdb seq, the pdb chain,
     the number of mismtaches and matches
    """
    maxMisMatches = max(5, default(maxMisMatches, int(0.1 * len(sequence))))
    if not os.path.isfile(pdbfile):
        # pylint: disable-next=broad-exception-raised
        raise Exception("ERROR: the pdb file does not exist: ", pdbfile)

    # extract PDB sequences by chains
    pdbseqs, residueLists, chains = extract_pdb_seq_from_pdb_file(
        pdbfile, chain_id=chain_id
    )

    bestPDBSeq = None
    bestMapping = None
    bestResidueList = None
    bestChain = None
    minMisMatches = np.iinfo(np.int32).max
    maxMatches = np.iinfo(np.int32).min
    for pdbseq, residueList, chain in zip(pdbseqs, residueLists, chains):
        if chain.get_id() != default(chain_id, chain.get_id()):
            # skip over chains without specified id
            continue

        seq2pdb_mapping, numMisMatches, numMatches = map_seq_to_residue_list(
            sequence, pdbseq, residueList
        )
        if seq2pdb_mapping is None:
            continue
        if maxMatches < numMatches:
            # if numMisMatches < minMisMatches:
            bestMapping = seq2pdb_mapping
            minMisMatches = numMisMatches
            maxMatches = numMatches
            bestResidueList = residueList
            bestPDBSeq = pdbseq
            bestChain = chain

    if minMisMatches > maxMisMatches:
        # print("Hola")
        print(
            f"ERROR: there are  {minMisMatches} mismatches between"
            f" the query sequence and PDB file: {pdbfile}\n"
            f"num residue : {len(sequence)}"
        )
        return None, None, None, None, None, None

    if maxMatches < min(30.0, minMatchRatio * len(sequence)):
        # print("Chau")
        print(
            "ERROR: there are only  {maxMatches} matches on query sequence, "
            f"less than  {minMatchRatio} of its length from PDB file: {pdbfile}"
        )
        return None, None, None, None, None, None

    return (
        bestMapping,
        bestResidueList,
        bestPDBSeq,
        bestChain,
        minMisMatches,
        maxMatches,
    )


def extract_coords_by_mapping(seq2pdb_mapping, residueList, atom_tys):
    """Extract coordinates from residue list by sequence mapping
    :param sequence:
    :param seq2pdb_mapping:
    :param residueList:
    :param atoms:
    :return:
    """
    needed_atoms = [a.upper() for a in atom_tys]
    needed_atom_set = set(needed_atoms)
    atomCoordinates = []
    for aligned_pos in seq2pdb_mapping:
        coordinates = defaultdict(lambda: None)

        if aligned_pos >= 0:
            res = residueList[aligned_pos]
            for atom in res:
                atom_name = atom.get_id().upper()
                if atom_name in needed_atom_set:
                    coordinates[atom_name] = atom.get_vector()

        atomCoordinates.append(coordinates)

    return atomCoordinates


def extract_coords_from_seq_n_pdb(
    sequence,
    pdbfile,
    atom_tys,
    chain_id: Optional[str] = None,
    maxMisMatches=5,
    minMatchRatio=0.5,
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
    """
    out = map_seq_to_pdb(
        sequence=sequence,
        pdbfile=pdbfile,
        chain_id=chain_id,
        maxMisMatches=maxMisMatches,
        minMatchRatio=minMatchRatio,
    )
    (
        seq2pdb_mapping,
        residueList,
        pdbseq,
        _,  # chain
        num_mismatches,
        num_matches,
    ) = out
    if seq2pdb_mapping is None:
        # print("Yes")
        return None, None, None, None, None
    residueList = list(residueList)
    atom_coordinates = extract_coords_by_mapping(
        seq2pdb_mapping=seq2pdb_mapping,
        residueList=residueList,
        atom_tys=atom_tys,
    )
    res_ids = [r.get_id()[1] for r in residueList]
    return atom_coordinates, pdbseq, num_mismatches, num_matches, res_ids


def listToTuple(function):
    def wrapper(*args, **kwargs):
        args = [tuple(x) if isinstance(x, list) else x for x in args]
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        result = function(*args, **kwargs)
        # result = tuple(result) if type(result) == list else result
        return result

    return wrapper


@listToTuple
@lru_cache(8)
def extract_atom_coords_n_mask_tensors(
    seq: Optional[str],
    pdb_path: str,
    atom_tys: List[str],
    chain_id: Optional[str] = None,
) -> Union[Tuple[Tensor, Tensor, Tensor, str], Tuple[Tensor, Tensor, str]]:
    """Extracts
    :param seq: sequence to map from
    :param pdb_path: pdb path to extract coordinates from
    :param atom_tys: atom types to extract coordinates for
    :return: Tuple containing
        (1) coords: Tensor of shape (n,a,3) containing atom coordinates for
        each atom type (1..a) and each residue 1..n in the input sequence
        (2) mask: Tensor of shape (n,a) indicating whether valid coordinates
        were obtained for atom types for each atom in atom_tys (1..a) and each
        residue (1..n) in the input sequence
    """
    assert seq is not None, "must provide sequence to extract coords and masks"
    out = extract_coords_from_seq_n_pdb(
        sequence=seq,
        pdbfile=pdb_path,
        atom_tys=atom_tys,
        chain_id=chain_id,
    )

    atom_coords, _, numMisMatches, *_ = out
    atom_mask = None
    #print(atom_coords is None)
    if atom_coords is not None:
        if numMisMatches > 5:
            print(
                f"WARNING: got {numMisMatches} ",
                f"mismatches mapping seq. to pdb\n{pdb_path}",
            )
        atom_coords, atom_mask = _get_coord_n_mask_tensors(atom_coords, atom_tys)
    return atom_coords, atom_mask


def _get_coord_n_mask_tensors(
    atom_coords: List[Dict[str, np.ndarray]], atom_tys: List[str]
) -> Tuple[Tensor, Tensor]:
    """Retrieves coord and mask tensors from output of
    extract_coords_from_seq_n_pdb(...).

    :param atom_coords: List of dictionaries. each dict mapping from atom type
      to atom coordinates.
    :param atom_tys: the atom types to extract coordinates for.
    :return: Tuple containing
        (1) coords: Tensor of shape (n,a,3) containing atom coordinates for
        each atom type (1...a) and each residue 1..n in the input sequence.
        (2) mask: Tensor of shape (n,a) indicating whether valid coordinates
        were obtained for atom types for each atom in atom_tys (1..a) and each
        residue (1...n) in the input sequence.
    """
    n_res, n_atoms = len(atom_coords), len(atom_tys)
    coords, mask = torch.zeros(n_res, n_atoms, 3), torch.zeros(n_res, n_atoms)
    for i, res in enumerate(atom_coords):
        for atom_pos, atom_ty in enumerate(atom_tys):
            if res[atom_ty] is None:
                continue
            coords[i, atom_pos] = torch.tensor([res[atom_ty][j] for j in range(3)])
            mask[i, atom_pos] = 1
    return coords, mask.bool()


def get_item_from_pdbs_n_seq(
    seq_paths: List[Optional[str]],
    decoy_pdb_paths: List[str],
    target_pdb_paths: List[Optional[str]],
    atom_tys: List[str],
    decoy_chain_ids: Optional[List[str]] = None,
    target_chain_ids: Optional[List[str]] = None,
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
    decoy_chain_ids = default(decoy_chain_ids, [None] * len(decoy_pdb_paths))
    target_chain_ids = default(target_chain_ids, [None] * len(target_pdb_paths)) # check!
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
        seq = safe_load_sequence(seq_path, decoy_pdb, chain_id=decoy_cid) # check! which sequence
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
        if decoy_data is None or target_data is None:
            return None  ##########################################################################################
        if default(tgt_pdb, decoy_pdb) == decoy_pdb:
            target_data = copy.deepcopy(target_data)

    # can't pickle a default dict
    batch["decoy"], batch["target"] = map(cast_defaultdict, (decoy_data, target_data))
    batch["decoy"]["pdb_paths"] = decoy_pdb_paths
    batch["target"]["pdb_paths"] = target_pdb_paths
    return batch


def cast_defaultdict(d):
    if isinstance(d, dict) or isinstance(d, defaultdict):
        return {k: cast_defaultdict(v) for k, v in d.items()}
    return d


def append_chain_to_data(
    data_dict: Dict,
    pdb_path: str,
    seq: str,
    atom_tys: Tuple[str],
    chain_id: Optional[str] = None,
) -> Dict:
    # target coords and mask
    crds, mask = extract_atom_coords_n_mask_tensors(
        seq, pdb_path=pdb_path, atom_tys=atom_tys, chain_id=chain_id
    )
    seq_encoding = torch.tensor([pc.AA_TO_INDEX[x] for x in seq])
    canonical_mask = aa_to_canonical_atom_mask(atom_tys)[seq_encoding]

    # print(mask is None, canonical_mask is None)
    try:  ##############################################################################################################################
        atom_mask = mask & canonical_mask
    except:
        return None
    ca_crds = repeat(
        crds[:, min(1, len(atom_tys)), :],
        "i c -> i a c",
        a=len(atom_tys),
    )
    crds[~atom_mask] = ca_crds[~atom_mask]

    data_dict["sequence"].append(seq)

    data_dict["tokenized_sequence"].append(seq_encoding.long())
    data_dict["atom_mask"].append(atom_mask)

    data_dict["coordinates"].append(crds)
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


@lru_cache(16)
def safe_load_sequence(
    seq_path: Optional[str], pdb_path: str, chain_id: Optional[str] = None
) -> str:
    """Loads sequence, either from fasta or given pdb file"""
    if exists(seq_path):
        pdbseqs = [load_fasta_file(seq_path)]
    else:
        # print(pdb_path)
        pdbseqs, *_ = extract_pdb_seq_from_pdb_file(pdb_path, chain_id=chain_id)
        # print(pdbseqs)
    if len(pdbseqs) > 1 and not exists(chain_id):
        print(f"[WARNING]: Multiple chains found for pdb: {pdb_path}")
    return pdbseqs[0]


def collate(batch: List[Optional[Dict]]) -> List[Dict]:
    return list(filter(exists, batch))


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
