# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a PyTorch learning repository containing notebooks, tutorials, and reference documentation organized around PyTorch and PyTorch Lightning. It is a personal knowledge base, not a library or application.

## Repository Structure

```
├── code/
│   ├── tutorials/              # Sequential Jupyter notebooks (01–13) covering PyTorch basics to transfer learning
│   │   ├── cnn_models/         # CNN implementations (ResNet)
│   │   ├── pytorch_lightning/  # PyTorch Lightning + W&B tutorials
│   │   └── vae/                # VAE notebooks
│   └── topics/
│       └── vae/                # Topic-deep-dives (anomaly detection with VAE)
├── docs/                       # Markdown reference docs
│   ├── lr-scheduler.md         # Comprehensive LR scheduler guide
│   ├── pytorch-lightning/      # PL patterns, W&B integration
│   └── models/                 # Model architecture docs (Autoencoder, VAE)
├── daily_knowledge.md          # Cumulative daily learning notes
└── assets/img/                 # Images referenced in docs
```

## Mac GPU Usage

On Apple Silicon, use MPS instead of CUDA:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## PyTorch Lightning Code Pattern

All Lightning models follow this required structure:

```python
class MyModel(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()   # saves hparams + auto-logs to W&B

    def forward(self, batch): ...
    def training_step(self, batch, batch_idx): ...
    def test_step(self, batch, batch_idx): ...
    def configure_optimizers(self): ...   # returns ([opt], [scheduler])
    def train_dataloader(self): ...
    def test_dataloader(self): ...
```

- Use `self.log(..., on_step=True, prog_bar=True, logger=True)` inside `training_step`
- `self.training` is inherited from `nn.Module` — Lightning sets it automatically during `fit()`/`test()`
- Load from checkpoint: `MyModel.load_from_checkpoint('path.ckpt')`; call `.freeze()` for inference

## Loss Function Conventions

- **Binary classification**: use `nn.BCEWithLogitsLoss` (raw logits, no sigmoid in `forward`). Apply `torch.sigmoid()` manually only at inference.
- **Multi-class classification**: use `nn.CrossEntropyLoss` (raw logits, no softmax in `forward`). Equivalent to `LogSoftmax + NLLLoss`.
- **VAE reconstruction loss**: use `F.smooth_l1_loss` (Huber loss, robust to outliers). Total loss = `recon_loss + kld_beta * kld`.

## LR Scheduler Key Rules

1. Always call `optimizer.step()` **before** `scheduler.step()`.
2. Always set `min_lr` / `eta_min` to prevent LR decaying to zero.
3. `OneCycleLR` must call `scheduler.step()` **per batch**, not per epoch.
4. For embedding-heavy or transformer models, use warmup: `SequentialLR(LinearLR → CosineAnnealingLR)`.
5. When combining `ReduceLROnPlateau` with early stopping: set early stopping patience ≥ 2× scheduler patience.

## W&B Integration

```python
pip install -q pytorch-lightning wandb
```

Pass `WandbLogger` to `pl.Trainer(logger=wandb_logger)`. Call `self.save_hyperparameters()` in `__init__` to auto-log hyperparameters. Close runs with `wandb.finish()` in notebooks.
