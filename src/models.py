from dataclasses import dataclass

from einops import rearrange, reduce, repeat
import numpy as np
import scipy as sp
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.blocks import DecoderBlock, ScaleNorm


@dataclass
class MaskformerConfig:

    query_dim: int
    low_res_feat_dim: int
    high_res_feat_dim: int
    num_transformer_layers: int

    mask_loss_type: str
    mask_loss_weight: float
    class_loss_weight: float

    def __post_init__(self):
        assert self.mask_loss_type in ("bce", "dice", "brier")


class Maskformer(nn.Module):

    def __init__(
        self,
        nn_module: nn.Module,
        num_queries: int,
        num_classes: int,
        config: MaskformerConfig,
    ):
        super().__init__()

        self.nn_module = nn_module
        self.num_queries = num_queries
        self.num_classes = num_classes

        self.query_dim = config.query_dim
        self.low_res_feat_dim = config.low_res_feat_dim
        self.high_res_feat_dim = config.high_res_feat_dim
        self.num_transformer_layers = config.num_transformer_layers

        self.mask_loss_type = config.mask_loss_type
        self.mask_loss_weight = config.mask_loss_weight
        self.class_loss_weight = config.class_loss_weight

        self.query_embed = nn.Parameter(torch.empty((self.num_queries, self.query_dim)))
        nn.init.xavier_normal_(self.query_embed)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(self.query_dim, self.query_dim, memory_dim=self.high_res_feat_dim)
        for _ in range(self.num_transformer_layers)])
        self.ln = ScaleNorm(self.query_dim)

        self.mask_embed_convs = nn.Sequential(
            nn.Linear(self.query_dim, self.query_dim),
            nn.ReLU(),
            nn.Linear(self.query_dim, self.query_dim),
            nn.ReLU(),
            nn.Linear(self.query_dim, self.low_res_feat_dim),
        )
        self.class_convs = nn.Sequential(
            nn.Linear(self.query_dim, self.query_dim),
            nn.ReLU(),
            nn.Linear(self.query_dim, self.query_dim),
            nn.ReLU(),
            nn.Linear(self.query_dim, self.num_classes + 1),
        )

    @staticmethod
    def preprocess_gt(y):
        #
        # Inputs:
        #   y: (bsz, tsz, h, w) integers where each entry denotes class index
        # Returns:
        #   gt_masks: (bsz, tsz, h, w) floats
        #   gt_classes: (bsz, tsz) longs
        #
        gt_masks = (y > 0).float()
        gt_classes = reduce(y, "b c h w -> b c", "max").long()
        return gt_masks, gt_classes

    def compute_loss_helper(self, gt_masks, gt_classes, pred_masks, pred_classes):
        #
        # We use the same loss for Hungarian matching as we do for computing loss for backprop.
        #
        # Inputs:
        #   gt_masks: (bsz, tsz, h, w)
        #   gt_classes: (bsz, tsz)
        #   pred_masks: (bsz, tsz, h, w)
        #   pred_classes: (bsz, tsz, num_classes)
        # Returns:
        #   loss: (bsz, tsz)
        #
        bsz, tsz, _, _ = gt_masks.shape

        class_loss = F.cross_entropy(
            pred_classes.view(bsz * tsz, -1),
            gt_classes.view(bsz * tsz),
            reduction="none",
        )
        class_loss = class_loss.view(bsz, tsz)

        if self.mask_loss_type == "bce":
            mask_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction="none")
            mask_loss = mask_loss.mean(dim=(-1, -2))
        elif self.mask_loss_type == "brier":
            mask_loss = 0.5 * ((pred_masks.sigmoid() - gt_masks) ** 2)
            mask_loss = mask_loss.mean(dim=(-1, -2))
        elif self.mask_loss_type == "dice":
            pred_masks_sigmoid = pred_masks.sigmoid()
            numerator = 2 * (pred_masks_sigmoid * gt_masks).sum(dim=(-1, -2))
            denominator = pred_masks_sigmoid.sum(dim=(-1, -2)) + gt_masks.sum(dim=(-1, -2))
            mask_loss = 1 - (numerator + 1) / (denominator + 1)
        else:
            raise AssertionError(f"Invalid {self.mask_loss_type=}.")

        return self.class_loss_weight * class_loss + self.mask_loss_weight * mask_loss

    @torch.no_grad()
    def compute_hungarian_matching(self, gt_masks, gt_classes, pred_masks, pred_classes):
        #
        # This implementation iterates over batch dimension to construct pairwise cost matrix,
        # which makes it less memory intensive.
        #
        # Takes space (tsz, tsz, h, w) rather than (bsz, tsz, tsz, h, w)
        #
        # Example:
        #   bsz = 64, tsz = 32, h = 240, w = 320 => ~0.6 GB assuming FP32
        #   bsz = 64, tsz = 16, h = 32, w = 32   => ~4.0 MB assuming FP32
        #
        bsz, tsz, _, _ = gt_masks.shape
        cost_matrix = torch.zeros((bsz, tsz, tsz), dtype=pred_classes.dtype)

        # Index (i, j) denotes cost of matching gt i to prediction j
        for idx in range(bsz):
            one_gt_classes = repeat(gt_classes[idx], "u -> u v", v=tsz).contiguous()
            one_pred_classes = repeat(pred_classes[idx], "v c -> u v c", u=tsz).contiguous()
            one_gt_masks = repeat(gt_masks[idx], "u h w -> u v h w", v=tsz).contiguous()
            one_pred_masks = repeat(pred_masks[idx], "v h w -> u v h w", u=tsz).contiguous()
            cost_matrix[idx] = self.compute_loss_helper(
                one_gt_masks,
                one_gt_classes,
                one_pred_masks,
                one_pred_classes,
            )

        # Hungarian matching needs to be done on CPU
        # Note that row_ind is always in order, col_ind is the only one permuted
        cost_matrix_numpy = cost_matrix.cpu().numpy()
        row_ind, col_ind = [], []
        for idx in range(bsz):
            one_row_ind, one_col_ind = sp.optimize.linear_sum_assignment(cost_matrix_numpy[idx])
            row_ind.append(one_row_ind)
            col_ind.append(one_col_ind)
        row_ind = torch.from_numpy(np.stack(row_ind))
        col_ind = torch.from_numpy(np.stack(col_ind))
        return row_ind, col_ind

    def loss(self, x, y):
        gt_masks, gt_classes = self.preprocess_gt(y)
        pred_masks, pred_classes = self.forward(x)

        _, col_ind = self.compute_hungarian_matching(gt_masks, gt_classes, pred_masks, pred_classes)
        pred_masks = torch.take_along_dim(pred_masks, col_ind.unsqueeze(-1).unsqueeze(-1), dim=1)
        pred_classes = torch.take_along_dim(pred_classes, col_ind.unsqueeze(-1), dim=1)

        loss = self.compute_loss_helper(gt_masks, gt_classes, pred_masks, pred_classes)
        return loss

    def forward(self, x):
        """
        Returns
        -------
            masks:   (bsz, # queries, h, w)
            classes: (bsz, # queries, # classes)
        """
        bsz = len(x)
        low_res_feat, high_res_feat = self.nn_module(x)
        low_res_feat = rearrange(low_res_feat, "b c h w -> b (h w) c")

        # Transformer blocks attend to the low resolution feature
        queries = repeat(self.query_embed, "t c -> b t c", b=bsz)
        for block in self.decoder_blocks:
            queries = block(queries, memory=low_res_feat)
        queries = self.ln(queries)

        # Predicted masks formed via per-pixel dot product w high resolution feature
        mask_embed = self.mask_embed_convs(queries)
        masks = torch.einsum("b t c, b c h w -> b t h w", mask_embed, high_res_feat)

        # Predicted classes via straightforward MLP
        classes = self.class_convs(queries)

        return masks, classes
