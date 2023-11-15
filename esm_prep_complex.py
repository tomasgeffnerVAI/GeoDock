import os
from multiprocessing import Pool
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from pathlib import Path

three_to_one = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",  # this is almost the same AA as MET. The sulfur is just replaced by Selen
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",
    "SER": "S",
    "SEC": "U",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "XAA": "X",
    "XLE": "J",
}


def process_file(path_):
    chain_to_seq = {}

    if ".pdb" not in str(path_):
        return chain_to_seq
        
    file_ = path_.stem
    pdb_id, _, _, _ = file_.split("_")
    #pdb_id = file_.replace(".pdb","")
    

    parser = PDBParser()
    structure = parser.get_structure("structure", path_)
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            residues = chain.get_residues()
            seq = []

            for residue in residues:
                residue_name = residue.get_resname()
                try:
                    seq_symbol = three_to_one[residue_name]
                    seq.append(seq_symbol)
                except KeyError:
                    pass  # Skip non-standard residues

            id_chain = f"{pdb_id}_{chain_id}"
            sequence = "".join(seq)
            chain_to_seq[id_chain] = sequence
            
    return chain_to_seq


def convert_to_fasta(file_names_w_dir, out_file):
    """
    Converts the protein sequences in the data_dir to a fasta file
    """
    file_names_w_dir = [file for file in Path(data_dir).iterdir() if file.is_file()]

    with Pool() as p:
        list_chain_to_seq = list(tqdm(p.imap(process_file, file_names_w_dir), total=len(file_names_w_dir)))

    chain_to_seq = {k: v for d in list_chain_to_seq for k, v in d.items()}

    records = []
    for index, seq in chain_to_seq.items():
        record = SeqRecord(Seq(seq), str(index))
        record.description = ""
        records.append(record)
    SeqIO.write(records, out_file, "fasta")

if __name__ == "__main__":
    data_dir = "data/test/test_single_chains"
    output_file = "data/fasta/test_single.fasta"
    convert_to_fasta(data_dir, output_file)
