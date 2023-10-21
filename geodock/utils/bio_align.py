from Bio import pairwise2
from Bio.pairwise2 import format_alignment


def align_seqs(seq_holo, seq_other):
    # sequence1, sequence2, match score, mismatch penalty, gap opening penalty, gap extension penalty
    alignments = pairwise2.align.globalms(seq_holo, seq_other, 2, -1, -5, -5)

    sorted_alignments = sorted(alignments, key=lambda x: x[2], reverse=True)

    # Get the best alignment
    best_alignment = sorted_alignments[0]

    # Extract the aligned sequences from the best alignment
    aligned_seq_holo, aligned_seq_other = best_alignment[0], best_alignment[1]

    return aligned_seq_holo, aligned_seq_other


# # Your protein residues as strings
# sequence1 = "MAGANDA"
# sequence2 = "MANA"

# # Align sequences using globalms
# # Match score is 1, mismatch is -1.
# # Gap opening penalty in sequence A is -1000 (to prevent gaps in sequence1)
# # Gap opening penalty in sequence B is -1.
# # Gap extension penalty in both sequences is -1.
# # alignments_xs = pairwise2.align.globalxs(sequence1, sequence2, 2, -1, -5)



# # Print the alignments
# for alignment in alignments:
#     print(format_alignment(*alignment))

# print(aligned_seq1, aligned_seq2)
