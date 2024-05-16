import argparse
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import MeteosatDataModule
from mthd.masking import MASK_STRATEGIES, MaskParams
from prestomodule import PrestoModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog_file_train', type=Path)
    parser.add_argument('--catalog_file_val', type=Path)
    parser.add_argument('--catalog_file_test', type=Path)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=Path, default=Path("lightning_logs"))
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--optimizer', '-o', type=str, default='sgd')
    parser.add_argument('--scheduler', '-sc', type=str, default='step')
    parser.add_argument('--compute_loss_lc', action='store_true')
    parser.add_argument('--positive_weight_loss_class',
                        type=float,
                        default=0.25)
    parser.add_argument('--lc_loss_weight', type=float, default=2.0)
    parser.add_argument(
        "--mask_strategies",
        type=str,
        default=[
            "group_bands",
            "random_timesteps",
            "chunk_timesteps",
            "random_combinations",
        ],
        nargs="+",
        help=
        "`all` will use all available masking strategies (including single bands)",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    return parser.parse_args()


def main():

    args = parse_args()
    print(f"epochs: {args.max_epochs}\n\
            bs: {args.batch_size}\n\
            lr: {args.lr}\n\
            optimizer: {args.optimizer}\n\
            scheduler: {args.scheduler}\n\
            compute_loss_lc: {args.compute_loss_lc}\n\
            lc_loss_weight: {args.lc_loss_weight}\n\
            mask_strategies: {args.mask_strategies}\n\
            mask_ratio: {args.mask_ratio}\n\
            positive_weight_loss_class: {args.positive_weight_loss_class}\n\
            log_dir: {args.log_dir}\n\
            seed: {args.seed}")
    assert args.optimizer in ["sgd", "adam"], "optimizer not supported"

    pl.seed_everything(args.seed)
    mask_strategies = tuple(args.mask_strategies)
    if (len(mask_strategies) == 1) and (mask_strategies[0] == "all"):
        mask_strategies = MASK_STRATEGIES
    mask_ratio = args.mask_ratio

    mask_params = MaskParams(mask_strategies, mask_ratio)
    data_module = MeteosatDataModule(args.catalog_file_train,
                                     args.catalog_file_val,
                                     args.catalog_file_test,
                                     batch_size=args.batch_size,
                                     mask_params=mask_params)

    model = PrestoModule(
        num_landcover_classes=data_module.num_lc_classes,
        band_groups=data_module.bands_groups,
        lr=args.lr,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        compute_loss_lc=args.compute_loss_lc,
        lc_loss_weight=args.lc_loss_weight,
        positive_weight=args.positive_weight_loss_class,
    )

    logger = TensorBoardLogger(
        args.log_dir,
        name=
        f"epochs_{args.max_epochs}_bs_{args.batch_size}_lr_{args.lr}_{args.optimizer}_{args.scheduler}_loss_lc_{args.compute_loss_lc}_lc_loss_weight_{args.lc_loss_weight}_seed_{args.seed}"
    )

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         devices=args.gpus,
                         logger=logger,
                         log_every_n_steps=10)

    trainer.fit(model, data_module)
    # test
    trainer.test(model, data_module)


main()
