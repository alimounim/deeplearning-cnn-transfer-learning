"""The CNN architecture and its LightningModuel wrapper."""
import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

def conv_block(c_in, c_out):
    """Conv -> BatchNorm -> ReLU -> MaxPool. Halves height and width."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class BasicCNN(nn.Module):
    """4 conv blocks (32-64-128-256 channels) + 2 fully connected layers."""

    def __init__(self, num_classes = 10, dropout = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3,32), # 128 -> 64
            conv_block(32, 64), # 64 -> 32
            conv_block(64, 128), # 32 -> 16
            conv_block(128, 256), # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1) # any input size -> 256 X 1 X 1
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout), # p = 0.0 means "off"
        )
        self.head = nn.Linear(128, num_classes) # the only task-specific layer

    def forward(self, x):
        return self.head(self.fc(self.pool(self.features(x))))

class ImageClassifier(L.LightningModule):
    def __init__(self, num_classes=10, dropout=0.0, lr=1e-3, weight_decay=1e-4):

        super().__init__()
        self.save_hyperparameters() # stored in every checkpoint
        self.model = BasicCNN(num_classes, dropout)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1)==y).float().mean()
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad] # skips frozen layers
        opt = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        return {"optimizer":opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


def from_pretrained(ckpt_path, num_classes=10, lr=3e-4, freeze_features=False):
    """Build a fresh classifier and copy every weight except the head from an Imagenette checkpoint."""
    pre = ImageClassifier.load_from_checkpoint(ckpt_path, map_location="cpu")
    model = ImageClassifier(num_classes=num_classes, dropout=pre.hparams.dropout, lr=lr)

    state = {k: v for k, v in pre.model.state_dict().items() if not k.startswith("head.")}
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    print("Not copied (new head):", missing)

    if freeze_features:
        for p in model.model.features.parameters():
            p.requires_grad = False
    return model

if __name__ == "__main__":
    model = ImageClassifier()
    x = torch.randn(2, 3, 128, 128)
    print(model(x).shape)
    print("params:", sum(p.numel() for p in model.parameters()))
