from argparse import ArgumentParser
import itertools
import logging

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.blocks import UNet
from src.models import Maskformer, MaskformerConfig


def gen_data(m, n, x_lim=32, y_lim=32, w=2, low=2, high=3):
    x = np.zeros((m, 1, y_lim, x_lim), dtype=np.int32)
    y = np.zeros((m, n, y_lim, x_lim), dtype=np.int32)
    for idx in range(m):
        n_instances = np.random.randint(low, high + 1)
        for k in range(n_instances):
            i = np.random.randint(w, y_lim - w)
            j = np.random.randint(w, x_lim - w)
            x[idx, :, i - w : i + w + 1, j - w : j + w + 1] = 1
            y[idx, k, i - w : i + w + 1, j - w : j + w + 1] = 1
    return x, y


def upsample_raster(raster, factor):
    h, w, c = raster.shape
    assert c == 3 and raster.dtype == np.uint8
    result = np.zeros((factor * h, factor * w, 3), dtype=np.uint8)
    for i, j in itertools.product(range(h), range(w)):
        result[i * factor : (i + 1) * factor, j * factor : (j + 1) * factor] = raster[i, j]
    return result


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", type=int, default=2000)
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--num-queries", type=int, default=8)
    argparser.add_argument("--device", type=str, default="cpu")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Hyper-parameters must be matched between (Maskformer <- UNet)
    #  (1) high_res_feat_dim = dim_scales[-1] * embed_dim => 128
    #  (2) low_res_feat_dim  = out_dim                    => 16
    nn_module = UNet(in_dim=1, out_dim=16, embed_dim=16, dim_scales=(1, 2, 4, 8))
    model = Maskformer(
        nn_module=nn_module,
        num_classes=1,
        num_queries=args.num_queries,
        config=MaskformerConfig(
            query_dim=64,
            high_res_feat_dim=128,
            low_res_feat_dim=16,
            num_transformer_layers=4,
            mask_loss_type="brier",
            mask_loss_weight=20.,
            class_loss_weight=1.,
        )
    )
    model = model.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations)

    normalize_fn = lambda t: 2 * (t - 0.5)  # [-1, +1]
    unnormalize_fn = lambda t: 0.5 * t + 0.5

    for i in range(args.iterations):

        x_tr, y_tr = gen_data(args.batch_size, args.num_queries)
        x_tr = torch.from_numpy(x_tr).to(args.device).float()
        x_tr = normalize_fn(x_tr)
        y_tr = torch.from_numpy(y_tr).to(args.device)
        optimizer.zero_grad()
        loss = model.loss(x_tr, y_tr).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 50 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.mean().data:.2f}\t")

    model.eval()

    x_te, y_te = gen_data(6, args.num_queries)
    x_te = torch.from_numpy(x_te).to(args.device).float()
    x_te = normalize_fn(x_te)

    with torch.no_grad():
        pred_masks, pred_classes = model.forward(x_te)

    x_te = unnormalize_fn(x_te).numpy()
    raster = np.zeros((6 * 32, (args.num_queries + 1) * 32), dtype=np.uint8)

    # Draw one column of GT followed by predictions
    for bsz_idx in range(6):
        raster[bsz_idx * 32 : (bsz_idx + 1) * 32, :32] = (
            (x_te[bsz_idx].squeeze(axis=0) * 255).astype(np.uint8)
        )
    for bsz_idx, tsz_idx in itertools.product(range(6), range(args.num_queries)):
        raster[bsz_idx * 32 : (bsz_idx + 1) * 32, (tsz_idx + 1) * 32 : (tsz_idx + 2) * 32] = (
            (torch.sigmoid(pred_masks[bsz_idx, tsz_idx]).data.numpy() * 255).astype(np.uint8)
        )

    # Convert to RGB
    raster = np.stack([raster, raster, raster], axis=-1)
    raster = upsample_raster(raster, factor=4)

    # Draw a simple grid
    for bsz_idx in range(6):
        raster[bsz_idx * 32 * 4, :] = 255
    for tsz_idx in range(8 + 1):
        raster[:, tsz_idx * 32 * 4] = 255

    pred_probs = F.softmax(pred_classes, dim=-1)[..., 1].round(decimals=2).numpy()
    print(pred_probs)
    plt.imsave("./examples/ex_squares.png", raster)

    import pdb; pdb.set_trace()
