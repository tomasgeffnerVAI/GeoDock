import os

# Define the folder containing the files
folder_path = "data/apo/apo_single_chains"

# Define the text file with filenames to be removed
text_file = "data/unprocessed/apo_unpr.txt"


# Read the list of filenames to remove from the text file
with open(text_file, "r") as file:
    filenames_to_remove = file.read().splitlines()

# Loop through the list of filenames, add the ".pdb" extension, and remove them from the folder
deleted_count = 0
for filename in filenames_to_remove:

    filename_ligand = filename + "_l_b.pdb"
    filename_receptor = filename + "_r_b.pdb"

    file_path_ligand = os.path.join(folder_path, filename_ligand)
    file_path_receptor = os.path.join(folder_path, filename_receptor)
    
    if os.path.exists(file_path_ligand):
        os.remove(file_path_ligand)
        deleted_count += 1
        print(f"Removed file: {file_path_ligand}")

    if os.path.exists(file_path_receptor):
        os.remove(file_path_receptor)
        deleted_count += 1
        print(f"Removed file: {file_path_receptor}")


print(f"{deleted_count} files were deleted.")
