import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils import data
from einops import repeat, rearrange
from geodock.datasets.matt_dataset import GeoDockDataset
from geodock.model.interface import GeoDockInput, GeoDockOutput
from geodock.model.modules.iterative_transformer import IterativeTransformer
from geodock.utils.loss import GeoDockLoss
from geodock.utils.crop import crop_features, crop_targets
from pytorch_lightning import Callback

from typing import Any, Dict



class GeoDock(pl.LightningModule):
    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 64,
        gm_depth: int = 1,
        sm_depth: int = 1,
        num_iter: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        # hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        # Embeddings
        esm_dim = 1280
        pair_dim = 65
        positional_dim = 69
        self.esm_to_node = nn.Linear(esm_dim, node_dim)
        self.pair_to_edge = nn.Linear(pair_dim, edge_dim)
        self.positional_to_edge = nn.Linear(positional_dim, edge_dim)

        # Networks
        self.net = IterativeTransformer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            gm_depth=gm_depth,
            sm_depth=sm_depth,
            num_iter=num_iter,
        )

        # Loss
        self.loss = GeoDockLoss()
        self.esm, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        for p in self.esm.parameters():
            p.requires_grad=False

    def forward(self, input: GeoDockInput):
        # Likely here call ESM embeddings on the fly
        # Stuff is batched / collated, so has an extra 1 in front of everything

        # print(type(input.esm_tokens))
        # print(input.esm_tokens.shape)
        # print(len(input.seq1[0]) + len(input.seq2[0]))
        # print(len(input.seq1[0]), len(input.seq2[0]))

        with torch.set_grad_enabled(False):
            ll = self.esm.num_layers
            reprs = self.esm(
                input.esm_tokens[0].long().to(self.device),
                repr_layers=[ll],
                return_contacts=False,
            )["representations"]
        # print(reprs[ll].shape)
        protein1_embeddings = reprs[ll][0, 1:-1, :]
        protein2_embeddings = reprs[ll][1, 1:-1, :]

        protein1_embeddings = protein1_embeddings[:len(input.seq1[0]), :][None, :, :]
        protein2_embeddings = protein2_embeddings[:len(input.seq2[0]), :][None, :, :]  # [L, 1280] -> [1, L, 1280]

        pair_embeddings = input.pair_embeddings
        positional_embeddings = input.positional_embeddings

        # print(protein1_embeddings.shape, protein2_embeddings.shape, pair_embeddings.shape, positional_embeddings.shape)

        protein1_embeddings, protein2_embeddings, pair_embeddings, positional_embeddings = crop_features(
            protein1_embeddings,
            protein2_embeddings,
            pair_embeddings,
            positional_embeddings,
        )

        # print(protein1_embeddings.shape, protein2_embeddings.shape, pair_embeddings.shape, positional_embeddings.shape)

        # Node embedding
        protein_embeddings = torch.cat(
            [protein1_embeddings, protein2_embeddings], dim=1
        )
        nodes = self.esm_to_node(protein_embeddings)

        # Edge embedding
        edges = self.pair_to_edge(pair_embeddings) + self.positional_to_edge(
            positional_embeddings
        )

        # Networks
        lddt_logits, dist_logits, coords, rotat, trans = self.net(
            node=nodes,
            edge=edges,
        )

        # Outputs
        output = GeoDockOutput(
            coords=coords,
            rotat=rotat,
            trans=trans,
            lddt_logits=lddt_logits,
            dist_logits=dist_logits,
        )

        return output

    def step(self, batch, batch_idx):
        # Get info from the batch
        # protein1_embeddings = batch["protein1_embeddings"]
        # protein2_embeddings = batch["protein2_embeddings"]
        pair_embeddings = batch["pair_embeddings"]
        positional_embeddings = batch["positional_embeddings"]
        seq1 = batch["seq1"]
        seq2 = batch["seq2"]

        # Prepare GeoDock input
        input = GeoDockInput(
            # protein1_embeddings=protein1_embeddings,
            # protein2_embeddings=protein2_embeddings,
            pair_embeddings=pair_embeddings,
            positional_embeddings=positional_embeddings,
            esm_tokens=batch["esm_tokens"],
            seq1=seq1,
            seq2=seq2,
        )

        # Get GeoDock output
        output = self(input)

        # print(batch["label_coords"].shape)
        # print(batch["label_rotat"].shape)
        # print(batch["label_trans"].shape)

        label_coords, label_rotat, label_trans, sep = crop_targets(
            batch["label_coords"],
            batch["label_rotat"],
            batch["label_trans"],
            L1=len(seq1[0]),
            L2=len(seq2[0]),
        )

        # print(label_coords.shape)
        # print(label_rotat.shape)
        # print(label_trans.shape)

        # Loss
        # losses = self.loss(output, batch)
        losses = self.loss(output, label_coords, label_rotat, label_trans, sep, self.training)
        intra_loss = losses["intra_loss"]
        inter_loss = losses["inter_loss"]
        dist_loss = losses["dist_loss"]
        lddt_loss = losses["lddt_loss"]
        violation_loss = losses["violation_loss"]

        if not self.training:
            # validation
            loss = intra_loss + inter_loss
        
        else:
            if self.current_epoch < 5:
                loss = intra_loss + 0.3 * dist_loss + 0.01 * lddt_loss
            elif self.current_epoch < 10:
                loss = intra_loss + inter_loss + 0.3 * dist_loss + 0.01 * lddt_loss
            else:
                loss = (
                    intra_loss
                    + inter_loss
                    + violation_loss
                    + 0.3 * dist_loss
                    + 0.01 * lddt_loss
                )

        losses.update({"loss": loss})

        return losses

    def _log(self, losses, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"{phase}/{loss_name}",
                indiv_loss,
                on_step=train,
                on_epoch=(not train),
                logger=True,
            )

            if train:
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )

    def training_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self._log(losses, train=True)

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self._log(losses, train=False)

        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.remove_esm_params(checkpoint)

    def remove_esm_params(self, checkpoint: Dict[str, Any]) -> None:
        # remove ESMEmbedding parameters
        del_keys = []
        for k in checkpoint["state_dict"]:
            if "ESMEmbedding.model" in k:
                del_keys.append(k)
        # log.info(f"Removing {len(del_keys)} keys for ESM from checkpoint")
        for k in del_keys:
            checkpoint["state_dict"].pop(k)


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """
    def on_before_optimizer_step(self, trainer, model, optim):
        gradient_norm(model)
    # def on_after_backward(self, trainer, model):
    #    model.log("info/clipped_grad_norm", gradient_norm(model))

def gradient_norm(model, check_missing: bool = True):
    total_norm = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        elif check_missing:
            print(f"grad none : {name}")
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


if __name__ == "__main__":
    dataset = GeoDockDataset()

    subset_indices = [0]
    subset = data.Subset(dataset, subset_indices)

    # load dataset
    dataloader = data.DataLoader(subset, batch_size=1, num_workers=6)

    model = GeoDock()
    trainer = pl.Trainer()
    trainer.validate(model, dataloader)
