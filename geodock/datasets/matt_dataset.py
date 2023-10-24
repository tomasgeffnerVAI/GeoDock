import os
import torch
import random
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
from einops import repeat

import sys

#sys.path.append("/home/celine/GeoDock")
# sys.path.append("/home/tomasgeffner/GeoDock")
from geodock.utils.pdb import save_PDB, place_fourth_atom
from geodock.utils.coords6d import get_coords6d
import numpy as np
from geodock.datasets.helpers import get_item_from_pdbs_n_seq
import geodock.datasets.protein_constants as pc
import pandas as pd


class GeoDockDataset(data.Dataset):
    def __init__(
        self,
        dataset: str = "pinder_toyexample_train",  # DQ why here and below?
        out_pdb: bool = False,
        out_png: bool = False,
        is_training: bool = True,
        is_testing: bool = False,
        prob: float = 1.0,
        count: int = 0,
        use_Cb: bool = True,
    ):
        self.skipped_n_hit = (0,0)
        self.dataset = dataset
        self.out_pdb = out_pdb
        self.out_png = out_png
        self.is_training = is_training
        self.is_testing = is_testing
        self.prob = prob
        self.count = count
        self.use_Cb = use_Cb


        # if dataset == "pinder_toyexample_train":
        #     self.data_dir = "/home/tomasgeffner/pinder_copy/splits_v2/train"
        #     self.data_list = "/home/tomasgeffner/pinder_copy/processed_train.txt"
        #     with open(self.data_list, "r") as f:
        #         lines = f.readlines()
        #     self.file_list = [line.strip() for line in lines]

        if dataset == "pinder_toyexample_test":
            self.data_dir = "/home/celine/pinder-public/splits/test"
            self.data_list = "/home/celine/pinder-public/test.txt"
            with open(self.data_list, "r") as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines]
        
        # if dataset == "pinder_toyexample_train":
        #     self.data_dir = "/home/celine/GeoDock_data/train"
        #     #self.file_list = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]  # TODO: clusters
        #     # L = len(self.file_list)  # These lines for training with 90%
        #     # self.file_list = self.file_list[:int(9 * L / 10)]
        #     cluster_file_path = "/home/celine/GeoDock_data/train_clusters.tsv"
        #     df =  pd.read_csv(cluster_file_path, sep="\t", header=None, names=["cluster", "pdbs"]) # todo pass cluster file
        #     self.file_list =df["cluster"].tolist()            

        
        # if dataset == "pinder_toyexample_val":
        #     self.data_dir = "/home/celine/GeoDock_data/train"
        #     self.file_list = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]  # TODO: clusters
        #     # L = len(self.file_list)  # These line for validation with 10%
        #     # self.file_list = self.file_list[int(9 * L / 10):]

        if dataset == "pinder_toyexample_train":
            self.data_dir = "/home/celine/pinder-public/splits/train"
            cluster_file_path = "/home/celine/pinder-public/train_clusters.tsv"
            df = pd.read_csv(cluster_file_path, sep="\t", header=None, names=["cluster", "pdbs"])  # todo pass cluster file
            cluster_list = df["cluster"].tolist()
            L = len(cluster_list)
            self.cluster_list = cluster_list[:int(9 * L / 10)]
            self.cluster_df = df

        
        if dataset == "pinder_toyexample_val":
            self.data_dir = "/home/celine/pinder-public/splits/train"
            cluster_file_path = "/home/celine/pinder-public/train_clusters.tsv"
            df = pd.read_csv(cluster_file_path, sep="\t", header=None, names=["cluster", "pdbs"])  # todo pass cluster file
            cluster_list = df["cluster"].tolist()
            L = len(cluster_list)
            self.cluster_list = cluster_list[int(9 * L / 10):]
            self.cluster_df = df

        # This to download: model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        _, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = alphabet.get_batch_converter()


    def __len__(self):
        return len(self.cluster_list)


    def get_decoy_receptor_ligand_pdbs(self, structure_root):
        # return pdb paths for receptor and ligand chain
        source_list = ['holo', 'apo', 'predicted']
        extension=["R.pdb", "L.pdb"]
        data_paths = dict()
        for ext in extension:
            random.shuffle(source_list)
            # for element in os.listdir(structure_root):
            for element in source_list:
                file_path = os.path.join(structure_root, element)
                
                # Just in case
                if not os.path.isdir(file_path):
                    continue

                # print(source_list, file_path)

                files_in_directory = os.listdir(file_path)
                if "element" == "apo":
                    if "alt" in files_in_directory:
                        files_in_directory = files_in_directory + os.listdir(os.path.join(file_path, "alt"))

                # check if file exists
                matching_files = [file for file in files_in_directory if file.endswith(ext)]
                random.shuffle(matching_files)
                if len(matching_files) > 0:
                    data_paths[ext] = os.path.join(file_path, matching_files[0])
                    break
        
        return data_paths["R.pdb"], data_paths["L.pdb"]


    def get_contiguous_crop(self, crop_len: int, n_res: int):
        """Get a contiguous interval to crop"""
        if crop_len < 0:
            return 0, n_res
        start, end = 0, n_res
        start = random.randint(0, (end - crop_len)) if end > crop_len else start
        return start, min(end, start + crop_len)
    
    
    def _get_item(self, idx: int):
        # cluster_file_path = "/home/celine/GeoDock_data/train_clusters.tsv"
        # #processed_file_path = 'graph_data/graphs' #if we want we can only include processed files
        # #processed_files = set({os.path.splitext(x)[0].replace("ligand_","").replace("receptor_",""): x for x in os.listdir(processed_file_path)[:200] if x.endswith(".bin")})
        # cluster_df = pd.read_csv(cluster_file_path, sep="\t", header=None, names=["cluster", "pdbs"]) # todo pass cluster file            
        # cluster_df["pdbs"] = cluster_df["pdbs"].apply(lambda x: [y for y in x.split(";")])
        # #cluster_df["pdbs"] = cluster_df["pdbs"].apply(lambda x: [y for y in x if y in processed_files])
        # cluster_df = cluster_df[cluster_df["pdbs"].apply(len) > 0]
        # self.file_list = [np.random.choice(x) for x in cluster_df['pdbs'] if x]
        # #print(len(self.file_list), self.file_list[0:3])
        
        cluster_name = self.cluster_list[idx]
        pdbs = self.cluster_df.loc[self.cluster_df['cluster'] == cluster_name, 'pdbs'].iloc[0]  # Each cluster has unique name this should be just a single element, hence the [0]
        pdbs = pdbs.split(";")
        _id = random.choice(pdbs)

        # print(cluster_name, len(pdbs), pdbs[0], "\n====")

        # # Get info from file_list
        # _id = self.file_list[idx]

        # load example
        structure_root = os.path.join(self.data_dir, _id)
        target_pdb = next(
           filter(lambda x: x.endswith("pdb"), os.listdir(structure_root))
        )
        target_pdb = os.path.join(structure_root,target_pdb)
        
        decoy_receptor_pdb, decoy_ligand_pdb = self.get_decoy_receptor_ligand_pdbs(
            structure_root
        )
        

        data = get_item_from_pdbs_n_seq(
            seq_paths=[None, None],
            decoy_pdb_paths=[decoy_receptor_pdb, decoy_ligand_pdb],
            target_pdb_paths=[target_pdb, target_pdb],
            # TODO: make atom types in same order as geodock!
            # atom_tys=tuple(pc.ALL_ATOMS),
            atom_tys=tuple(pc.BB_ATOMS_GEO),
            decoy_chain_ids=[None,None],  # TODO: might be B, A?? RL!
            target_chain_ids=["R","L"],
        )

        if data is None:
            return None  ####################################################################################################################################

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
        # TODO: Fix sequence for embeddings, we cannot mask here, not yet - NO

        #### generate input ####
        decoy_coords = torch.cat([coords1_decoy, coords2_decoy], dim=0)

        # Pair embedding
        input_pairs = self.get_pair_mats(decoy_coords, len(seq1))

        # Contact embedding
        if self.is_training:
            input_contact = self.get_pair_contact(
                decoy_coords, len(seq1), count=random.randint(0, 3)
            )
        else:
            input_contact = self.get_pair_contact(
                decoy_coords, len(seq1), count=self.count
            )

        pair_embeddings = torch.cat([input_pairs, input_contact], dim=-1)

        # Pair positional embedding
        positional_embeddings = self.get_pair_relpos(len(seq1), len(seq2))

        try:
            assert positional_embeddings.size(0) == pair_embeddings.size(0)
            assert positional_embeddings.size(1) == pair_embeddings.size(1)
        except:
            print(_id)

        #### generate target ####
        label_coords = torch.cat([coords1_true, coords2_true], dim=0)
        label_rotat = self.get_rotat(label_coords)
        label_trans = self.get_trans(label_coords)


        if self.out_pdb:
            # print(_id)
            test_coords = self.get_full_coords(label_coords)
            out_file = _id + ".pdb"
            if os.path.exists(out_file):
                os.remove(out_file)
                print(f"File '{out_file}' deleted successfully.")
            else:
                print(f"File '{out_file}' does not exist.")
            save_PDB(
                out_pdb=out_file,
                coords=test_coords,
                seq=seq1 + seq2,
                delim=len(seq1) - 1,
            )

        *_, tokens = self.batch_converter([("1", seq1), ("2", seq2)])

        # Output
        output = {
            "id": _id,
            "seq1": seq1,
            "seq2": seq2,
            "esm_tokens": tokens,
            "pair_embeddings": pair_embeddings,
            "positional_embeddings": positional_embeddings,
            "label_rotat": label_rotat,
            "label_trans": label_trans,
            "label_coords": label_coords,
        }

        return {key: value for key, value in output.items()}


    def __getitem__(self, idx: int):
        example = None
        skipped,hit = self.skipped_n_hit

        while example is None:
            try:
                example = self._get_item(idx)  # regular dataset __getitem__ function
            
            except Exception as e:
                print("__getitem__ returned None")
                print(e)
                example = None
            
            if example is None:
            
                idx = random.randint(0, len(self))
                skipped+=1
            else:
                #print([k for k in example])
                c1,c2 =len(example["seq1"]),len(example["seq2"])
                if min(c1,c2)<30:
                    example=None
                idx = random.randint(0, len(self))
        
        hit += 1
        self.skipped_n_hit=(skipped,hit)
        # print(skipped,hit)
        return example

    # def __getitem__(self, idx: int):
    #     example = None
    #     skipped,hit = self.skipped_n_hit
    #     while example is None:
            
    #         example = self._get_item(idx)
    #         idx = random.randint(0, len(self))
    #         if example is None:
    #             skipped+=1
            
    #     hit+=1
    #     self.skipped_n_hit=(skipped,hit)
    #     # print(skipped,hit)
    #     return example

    def get_rotat(self, coords):
        # Get backbone coordinates.
        n_coords = coords[:, 0, :]
        ca_coords = coords[:, 1, :]
        c_coords = coords[:, 2, :]

        # Gram-Schmidt process.
        v1 = c_coords - ca_coords
        v2 = n_coords - ca_coords
        e1 = F.normalize(v1)
        u2 = v2 - e1 * (torch.einsum("b i, b i -> b", e1, v2).unsqueeze(-1))
        e2 = F.normalize(u2)
        e3 = torch.cross(e1, e2, dim=-1)

        # Get rotations.
        rotations = torch.stack([e1, e2, e3], dim=-1)
        return rotations

    def get_trans(self, coords):
        return coords[:, 1, :]

    def get_pair_relpos(self, rec_len, lig_len):
        rmax = 32
        rec = torch.arange(0, rec_len)
        lig = torch.arange(0, lig_len)
        total = torch.cat([rec, lig], dim=0)
        pairs = total[None, :] - total[:, None]
        pairs = torch.clamp(pairs, min=-rmax, max=rmax)
        pairs = pairs + rmax
        pairs[:rec_len, rec_len:] = 2 * rmax + 1
        pairs[rec_len:, :rec_len] = 2 * rmax + 1
        relpos = F.one_hot(pairs, num_classes=2 * rmax + 2).float()
        total_len = rec_len + lig_len
        chain_row = torch.cat(
            [torch.zeros(rec_len, total_len), torch.ones(lig_len, total_len)], dim=0
        )
        chain_col = torch.cat(
            [torch.zeros(total_len, rec_len), torch.ones(total_len, lig_len)], dim=1
        )
        chains = F.one_hot((chain_row - chain_col + 1).long(), num_classes=3).float()

        pair_pos = torch.cat([relpos, chains], dim=-1)
        return pair_pos

    def get_pair_contact(self, coords, n, prob=None, count=None):
        assert (prob is None) ^ (count is None)

        if self.use_Cb:
            coords = self.get_full_coords(coords)[:, -1, :]
        else:
            coords = coords[:, 1, :]

        d = torch.norm(coords[:, None, :] - coords[None, :, :], dim=2)
        cutoff = 10.0
        mask = d <= cutoff

        mask[:n, :n] = False
        mask[n:, n:] = False

        rec = mask[:n, :n]
        lig = mask[n:, n:]
        inter = mask[:n, n:]

        if prob is not None:
            random_tensor = torch.rand_like(inter, dtype=torch.float)
            inter = inter & (random_tensor > prob)

        elif count is not None:
            # get the indices of all the True values in the tensor
            true_indices = torch.nonzero(inter, as_tuple=False)

            # shuffle the indices
            shuffled_indices = torch.randperm(true_indices.shape[0])

            # make sure count <= # of true indices
            if count > true_indices.shape[0]:
                count = true_indices.shape[0]

            # pick the first shuffled index
            selected_indices = true_indices[shuffled_indices[:count]]

            # create a new tensor of the same shape as the original tensor
            mask = torch.zeros_like(inter)

            # set the randomly selected indices to True
            for i in range(selected_indices.shape[0]):
                mask[selected_indices[i, 0], selected_indices[i, 1]] = True

            inter = inter * mask

        upper = torch.cat([rec, inter], dim=1)
        lower = torch.cat([inter.T, lig], dim=1)
        contact = torch.cat([upper, lower], dim=0)

        return contact.unsqueeze(-1)

    def get_pair_dist(self, coords, n):
        num_bins = 16
        distogram = self.distogram(
            coords,
            2.0,
            22.0,
            num_bins,
        )

        distogram[:n, n:] = num_bins - 1
        distogram[n:, :n] = num_bins - 1

        # to onehot
        dist = F.one_hot(distogram, num_classes=num_bins).float()

        # test
        if self.out_png:
            data = distogram.numpy()
            plt.imshow(data, cmap="hot", interpolation="nearest")
            plt.colorbar()

            # Save the plot as a PNG file
            plt.savefig("dist.png", dpi=300)
            plt.clf()

        return dist

    def get_pair_mats(self, coords, n):
        dist, omega, theta, phi = get_coords6d(coords, use_Cb=self.use_Cb)

        mask = dist < 22.0

        num_bins = 16
        dist_bin = self.get_bins(dist, 2.0, 22.0, num_bins)
        omega_bin = self.get_bins(omega, -180.0, 180.0, num_bins)
        theta_bin = self.get_bins(theta, -180.0, 180.0, num_bins)
        phi_bin = self.get_bins(phi, -180.0, 180.0, num_bins)

        def mask_mat(mat):
            mat[~mask] = num_bins - 1
            mat.fill_diagonal_(num_bins - 1)
            mat[:n, n:] = num_bins - 1
            mat[n:, :n] = num_bins - 1
            return mat

        dist_bin[:n, n:] = num_bins - 1
        dist_bin[n:, :n] = num_bins - 1
        omega_bin = mask_mat(omega_bin)
        theta_bin = mask_mat(theta_bin)
        phi_bin = mask_mat(phi_bin)

        # to onehot
        dist = F.one_hot(dist_bin, num_classes=num_bins).float()
        omega = F.one_hot(omega_bin, num_classes=num_bins).float()
        theta = F.one_hot(theta_bin, num_classes=num_bins).float()
        phi = F.one_hot(phi_bin, num_classes=num_bins).float()

        # test
        if self.out_png:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(nrows=2, ncols=2)
            ax[0, 0].imshow(dist_bin.numpy(), cmap="hot", interpolation="nearest")
            ax[0, 1].imshow(omega_bin.numpy(), cmap="hot", interpolation="nearest")
            ax[1, 0].imshow(theta_bin.numpy(), cmap="hot", interpolation="nearest")
            ax[1, 1].imshow(phi_bin.numpy(), cmap="hot", interpolation="nearest")

            # Set titles for each plot
            ax[0, 0].set_title("dist")
            ax[0, 1].set_title("omega")
            ax[1, 0].set_title("theta")
            ax[1, 1].set_title("phi")

            # Save the plot as a PNG file
            plt.savefig("dist_orient.png", dpi=300)
            plt.clf()

        return torch.cat([dist, omega, theta, phi], dim=-1)

    def get_full_coords(self, coords):
        # get full coords
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
        # print(coords.shape, C.shape, CA.shape)
        # Infer CB coordinates.
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

        O = place_fourth_atom(
            torch.roll(N, -1, 0),
            CA,
            C,
            torch.tensor(1.231),
            torch.tensor(2.108),
            torch.tensor(-3.142),
        )
        full_coords = torch.stack([N, CA, C, O, CB], dim=1)

        return full_coords

    def distogram(self, coords, min_bin, max_bin, num_bins, use_cb=False):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=coords.device,
        )
        boundaries = boundaries**2
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]

        if use_cb:
            # Infer CB coordinates.
            b = CA - N
            c = C - CA
            a = b.cross(c, dim=-1)
            CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
            dists = (
                (CB[..., None, :, :] - CB[..., :, None, :])
                .pow(2)
                .sum(dim=-1, keepdims=True)
            )
        else:
            dists = (
                (CA[..., None, :, :] - CA[..., :, None, :])
                .pow(2)
                .sum(dim=-1, keepdims=True)
            )

        bins = torch.sum(dists > boundaries, dim=-1)  # [..., L, L]
        return bins

    def get_bins(self, x, min_bin, max_bin, num_bins):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=x.device,
        )
        bins = torch.sum(x.unsqueeze(-1) > boundaries, dim=-1)  # [..., L, L]
        return bins


if __name__ == "__main__":
    dataset = GeoDockDataset(
        dataset="pinder_toyexample_train",
        out_pdb=True,
        out_png=False,
        is_training=False,
        count=0,
    )

    dataset[0]

    """
    dataloader = data.DataLoader(dataset, batch_size=1, num_workers=6)

    for batch in tqdm(dataloader):
        pass
    """
