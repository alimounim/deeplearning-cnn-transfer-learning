"""One entry point for every experiment in the assignment.

  python -m src.train --run basic                       # Part I
  python -m src.train --run regularized                 # Part II
  python -m src.train --run cifar_scratch               # Part III (baseline)
  python -m src.train --run cifar_finetune --ckpt PATH  # Part III (transfer)
"""
import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.datamodules import CIFAR10DataModule, ImagenetteDataModule
from src.models import ImageClassifier, from_pretrained


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True,
                   choices=["basic", "regularized", "cifar_scratch", "cifar_finetune"])
    p.add_argument("--ckpt", help="Imagenette checkpoint (for cifar_finetune)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-3 from scratch, 3e-4 when fine-tuning")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--freeze", action="store_true", help="freeze conv layers when fine-tuning")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--fast-dev-run", action="store_true", help="1 batch smoke test")
    return p.parse_args()


def build(args):
    common = dict(data_dir=args.data_dir, image_size=args.img_size,
                  batch_size=args.batch_size, num_workers=args.num_workers)
    lr = args.lr or 1e-3

    if args.run == "basic":
        dm = ImagenetteDataModule(augment=False, **common)
        model = ImageClassifier(dropout=0.0, lr=lr)
    elif args.run == "regularized":
        dm = ImagenetteDataModule(augment=True, **common)
        model = ImageClassifier(dropout=args.dropout, lr=lr)
    elif args.run == "cifar_scratch":
        dm = CIFAR10DataModule(augment=True, **common)
        model = ImageClassifier(dropout=args.dropout, lr=lr)
    else:  # cifar_finetune
        if not args.ckpt:
            raise SystemExit("--ckpt is required for cifar_finetune")
        dm = CIFAR10DataModule(augment=True, **common)
        # Smaller LR so early updates don't wreck the pretrained features
        model = from_pretrained(args.ckpt, lr=args.lr or 3e-4, freeze_features=args.freeze)
    return dm, model


def main():
    args = parse_args()
    L.seed_everything(42)
    dm, model = build(args)

    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                 filename="best-{epoch:02d}-{val_acc:.3f}")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",              # GPU if available
        logger=CSVLogger("logs", name=args.run),
        callbacks=[checkpoint, early_stop],
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, datamodule=dm)

    if args.fast_dev_run:
        return
    print("Best checkpoint:", checkpoint.best_model_path)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
