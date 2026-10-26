"""LightningDataModules for Imagenette and CIFAR-10"""
import lightning as L
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T

# ImageNet statistics: Imagenette is a subset of ImageNet, so these fit well.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transforms(image_size: int, augment: bool):
    """Return (train_transform, eval_transform)."""
    eval_tf = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if not augment:
        return eval_tf, eval_tf

    train_tf = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, eval_tf

class _BaseDataModule(L.LightningDataModule):
    """Shared logic: split the official train set into train/val, use the official held-out set as the test set. """
    def __init__(self, data_dir="data", batch_size=64, num_workers=4, image_size=128, augment=False, val_fraction=0.1, seed=42):
        super().__init__()
        self.save_hyperparameters()
        self.train_tf, self.eval_tf = build_transforms(image_size, augment)

    def _train_set(self, transform, download=False):
        raise NotImplementedError

    def _test_set(self, transform, download=False):
        raise NotImplementedError

    def prepare_data(self):
        # Runs once (on one process): download only
        self._train_set(None, download=True)
        self._test_set(None, download=True)

    def setup(self, stage=None):
        #Two copies of the same images: one with train transforms, one with eval transforms.
        train_aug = self._train_set(self.train_tf)
        train_plain = self._train_set(self.eval_tf)

        n = len(train_aug)
        n_val = int(n * self.hparams.val_fraction)
        g = torch.Generator().manual_seed(self.hparams.seed)
        idx = torch.randperm(n, generator=g).tolist()

        self.train_ds = Subset(train_aug, idx[n_val:])
        self.val_ds = Subset(train_plain, idx[:n_val]) # no augmentation on validation

        self.test_ds = self._test_set(self.eval_tf)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=shuffle, num_workers=self.hparams.num_workers, persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)


class ImagenetteDataModule(_BaseDataModule):
    """10 ImageNet classes; ~9.5k train / ~3.9k val images. we use the 160px version."""
    num_classes = 10

    def _train_set(self, transform, download=False):
        return datasets.Imagenette(self.hparams.data_dir, split="train", size="160px", download=download, transform=transform)

    def _test_set(self, transform, download=False):
        return datasets.Imagenette(self.hparams.data_dir, split="val", size="160px", download=download, transform=transform)


class CIFAR10DataModule(_BaseDataModule):
    """10 classes; 50k train / 10k test images, 32x32. Upscaled to img_size so the
    pretrained Imagenette filters see objects at a similar scale."""
    num_classes = 10

    def _train_set(self, transform, download=False):
        return datasets.CIFAR10(self.hparams.data_dir, train=True,
                                download=download, transform=transform)

    def _test_set(self, transform, download=False):
        return datasets.CIFAR10(self.hparams.data_dir, train=False,
                                download=download, transform=transform)


if __name__ == "__main__":
    dm = ImagenetteDataModule(augment=True)
    dm.prepare_data()
    dm.setup()
    print("train/val/test:", len(dm.train_ds), len(dm.val_ds), len(dm.test_ds))
    x, y = next(iter(dm.train_dataloader()))
    print("batch:", x.shape, "labels:", y[:10])



