# Pytorch Lightning Introduction

## Introduction

- Pytorch Lightning is a lightweight wrapper for organizing your PyTorch code and easily adding advanced features such as distributed training, 16-bit precision or gradient accumulation.

### Pytorch Lightning with Weights & Biases integration

- [Weights & Biases integration](https://docs.wandb.com/library/integrations/lightning) with Pytorch Lightning for full traceability and reproducibility with only extra lines of code:

```python
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from collections import OrderedDict
hparams = OrderedDict(
    run='embsz16_latsz16_bsz128_lay64-128-256-128-64_ep100_cosineWR', # define the run name with the model parameters for the ease of tracking
    version=1,
    project_name='VAE_Anomaly'
)

wandb_logger = WandbLogger(
    project=hparams['project_name'],
    name=hparams['run'],
    version=hparams['version']
) # init Wandb logger
trainer = Trainer(logger=wandb_logger) # provide the Wandb logger into Pl's Trainer class
```

- W&B integration with Pytorch-Lightning can automatically:
  - log your configuration parameters
  - log your losses and metrics
  - log your model
  - keep track of your code
  - log your system metrics (GPU, CPU, memory, temperature, etc)
- Read more on:
  - PyTorch Lightning WandbLogger docs [here](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.loggers.WandbLogger.html?highlight=wandblogger)
  - Weights & Biases docs [here](https://docs.wandb.com/library/integrations/lightning)

### Installation

```Shell
pip install -q pytorch-lightning wandb
```

- Logging into W&B so that our experiments can be associated with our account by running below code snippet and provide the Wandb's API key:

```Python
import wandb
wandb.login()
```

- [For Kaggle Notebook Only]
  - Add the Wandb's API key into the Kaggle's Secret in `Add-ons` tab with key is `WANDB_API_KEY` and value is the Wandb's API key
  - Retreive the Kaggle secret via `kaggle_secrets` package
  ```Python
  from kaggle_secrets import UserSecretsClient
  user_secrets = UserSecretsClient()
  wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
  ```
  - Run the `wandb login` via CLI
  ```shell
  wandb login $wandb_api_key
  ```

## Pytorch Lightning Code Structure

### Creating `Dataset` Class

- Prepare the dataset class that is able to load either `train` or `test` data via `split` variable

```Python
from torch.utils.data import Dataset

class MnistDataset(Dataset):
    def __init__(self,
                 split,
                 data_folder_path,
                 cont_vars=None,
                 cat_vars=None,
                 target_var=None,
                ):
        """
        split: 'train' if we want to get data from the training examples, 'test' for
        test examples, or 'both' to merge the training and test sets and return samples
        from either.
        data_folder_path: Folder path contains data files

        """
        super().__init__()
        assert split in ['train', 'test', 'both']

        if split == 'train':
            self.df = pd.read_csv(data_folder_path/'train.csv')
        elif split == 'test':
            self.df = pd.read_csv(data_folder_path/'test.csv')
        else:
            df1 = pd.read_csv(data_folder_path/'train.csv')
            df2 = pd.read_csv(data_folder_path/'test.csv')
            self.df = pd.concat((df1, df2), ignore_index=True)

        # convert continous columns into numpy array
        self.cont = self.df[cont_vars].copy().to_numpy(dtype=np.float32)
        # convert encoded categorical columns into numpy array
        self.cat = self.df[cat_vars].copy().to_numpy(dtype=np.int64)
        # convert target column into numpy array
        self.target = self.df[target_var].copy().to_numpy(dtype=np.float32)

    def __getitem__(self, idx):
        """
        This to return the 3 tensors:
            - An tensor (array) of continuous values represented by floats
            - Categorical variable tensor
            - Target variable tensor
        """
        return torch.tensor(self.cont[idx]), torch.tensor(self.cat[idx]), torch.tensor(self.target[idx])

    def __len__(self):
        return self.df.shape[0]

# to view the dataset
ds = MnistDataset(split='both',
                  data_folder_path=output_path,
                  cont_vars=['value', 't'],
                  cat_vars=['day_of_week', 'holiday'],
                  target_var=['target'])
print(len(ds))

it = iter(ds)
for _ in range(10):
    # print the first 10 items in the dataset
    # print(next(it))
    cont_features, cat_features, target = next(it)
```

### Defining the Model

- **Tips**:
  - Call `self.save_hyperparameters()` in `__init__`
    - to save all the arguments passed to the constructor (`__init__`) as hyperparameters into the dictionary `self.hparams`
    - to automatically log your hyperparameters to **W&B**
  - Call `self.log(on_step=True)` in `training_step` and `validation_step` to log the metrics
    - `on_step=True` the metric will be logged at every training step. Conversely, when `on_step=False`, the metric will be logged at the end of an epoch.
  - `self.training` is an attribute that is inherited from Pytorch's `nn.Module` and it returns `True` if the module is in training mode (`model.train()`) and `False` if it's in evaluation mode (`model.eval()`)
    - Note: In Pytorch Lightning, when you call `trainer.fit()`, PyTorch Lightning internally handles setting the model to training mode (`model.train()`) during the training loop. Conversely, when you call `trainer.test()`, it sets the model to evaluation mode (`model.eval()`) for the evaluation loop.

```Python
import pytorch_lightning as pl
from collections import OrderedDict

hparams = OrderedDict(
    run='embsz16_latsz16_bsz128_lay64-128-256-128-64_ep100_cosineWR', # define the run name with the model parameters for the ease of tracking
    version=1,
    project_name='VAE_Anomaly',
    cont_vars = cont_features,
    cat_vars = cat_features,
    embedding_sizes = [(embed_cats[i], 16) for i in range(len(embed_cats))],
    latent_dim = 16,
    layer_sizes = '64,128,256,128,64',
    batch_norm = True,
    stdev = 0.1,
    kld_beta = 0.05,
    lr = 0.001,                     # Learning Rate
    weight_decay = 1e-5,            # Weight Decay
    batch_size = 128,               # Batch Size
    epochs = 60,                    # Epoch
    data_folder_path=output_path,
)
# define a single layer as a sequence of {fully-connected unit, batch normalisation, leaky-ReLU activation}
class Layer(nn.Module):
    '''
    A single fully connected layer with optional batch
    normalisation and activation.
    '''
    def __init__(self, in_dim, out_dim, bn = True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if bn: layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    '''
    The decoder part of our VAE. Takes a latent vector (sampled from the distribution learned by the
    encoder) and converts it back to a reconstructed data sample.
    '''
    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)

        hidden_dims = [self.hparams.latent_dim] + [int(s) for s in reversed(self.hparams.layer_sizes.split(','))]
        out_dim = sum(emb_size for _, emb_size in self.hparams.embedding_sizes) + len(self.hparams.cont_vars)
        bn = self.hparams.batch_norm

        self.layers = nn.Sequential(
            *[Layer(hidden_dims[i], hidden_dims[i + 1], bn) for i in range(len(hidden_dims) - 1)],
        )
        self.reconstructed = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, z):
        h = self.layers(z)
        recon = self.reconstructed(h)
        return recon

class VAE(pl.LightningModule):
    def __init__(self, **hparams):
        """
        [Required] This is to init the Pl's Lightning module
        """
        super().__init__()

        self.save_hyperparameters() # to save **hparams in self.hparams attribute
        self.encoder = Encoder(**hparams) # NN's blocks defined based on the nn.Module
        self.decoder = Decoder(**hparams) # NN's blocks defined based on the nn.Module

    def _reparameterize(self, mu, logvar):
        """
        [Optional] This is extra method to cater for this VAE model (apart from the required methods in Pl's model)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * self.hparams.stdev
            return eps * std + mu
        else:
            return mu

    def forward(self, batch):
        """
        [Required] This defines model's forward method
        """
        x_cont, x_cat, y = batch
        assert x_cat.dtype == torch.int64
        # encoder
        mu, logvar, x = self.encoder(x_cont, x_cat)
        # re-parameterise
        z = self._reparameterize(mu, logvar)
        # decoder
        recon = self.decoder(z)

        return recon, mu, logvar, x

    def loss_function(self, ground_truth, preds, mu, logvar):
        """
        [Optional] This is required only if there is a custom loss required
        """
        recon_loss = F.smooth_l1_loss(preds, ground_truth, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        return recon_loss, kld

    def training_step(self, batch, batch_idx):
        """
        [Required] This defines the training steps
        """
        recon, mu, logvar, x = self.forward(batch)
        # The loss function compares the concatenated input vector including
        # embeddings to the reconstructed vector
        recon_loss, kld = self.loss_function(x, recon, mu, logvar)

        loss = recon_loss + self.hparams.kld_beta * kld

        self.log('total_loss', loss.mean(dim=0), on_step=True, prog_bar=True,
                 logger=True)
        self.log('recon_loss', recon_loss.mean(dim=0), on_step=True, prog_bar=True,
                 logger=True)
        self.log('kld', kld.mean(dim=0), on_step=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        [Required] This defines the test steps
        """
        recon, mu, logvar, x = self.forward(batch)
        recon_loss, kld = self.loss_function(x, recon, mu, logvar)
        loss = recon_loss + self.hparams.kld_beta * kld
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        """
        [Required] This defines the optimizer
        """
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay,
                                eps=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=25, T_mult=1, eta_min=1e-9, last_epoch=-1)
        return [opt], [sch]

    def train_dataloader(self):
        """
        [Required] This defines the train dataloader
        """
        dataset = TsDataset('train', data_folder_path=self.hparams.data_folder_path, cont_vars=self.hparams.cont_vars,
            cat_vars = self.hparams.cat_vars, lbl_as_feat=True
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=2,
            pin_memory=True, persistent_workers=True, shuffle=True
        )

    def test_dataloader(self):
        """
        [Required] This defines the test dataloader
        """
        dataset = TsDataset('test', data_folder_path=self.hparams.data_folder_path, cont_vars=self.hparams.cont_vars,
            cat_vars=self.hparams.cat_vars, lbl_as_feat=True
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=2,
            pin_memory=True, persistent_workers=True
        )

# init the model with hparams
model = VAE(**hparams)
```

### Defining Callbacks & `WandbLogger`

- The Pl's `ModelCheckpoint` callback is required along with the `WandbLogger` argument to log model checkpoints to W&B.

```Python
from pytorch_lightning.callbacks import ModelCheckpoint
# define model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='test_loss',
    mode='min',
    dirpath='./model_checkpoints',
    filename='vae_weights')
```

- PyTorch Lightning has a `WandbLogger` to easily log your experiments with Wights & Biases by providing it to `Trainer` object.
  - See the [WandbLogger docs](https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.WandbLogger.html#pytorch_lightning.loggers.WandbLogger) for all parameters.

| Functionality                                         | Argument/Function                                                               | PS                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Logging models                                        | `WandbLogger(... ,log_model='all')` or `WandbLogger(... ,log_model=True`)       | Log all models if `log_model="all"` and at end of training if `log_model=True` |
| Set custom run names                                  | `WandbLogger(... ,name='my_run_name'`)                                          |                                                                                |
| Organize runs by project                              | `WandbLogger(... ,project='my_project')`                                        |                                                                                |
| Log histograms of gradients and parameters            | `WandbLogger.watch(model)`                                                      | `WandbLogger.watch(model, log='all')` to log parameter histograms              |
| Log hyperparameters                                   | Call `self.save_hyperparameters()` within `LightningModule.__init__()`          |
| Log custom objects (images, audio, video, molecules…) | Use `WandbLogger.log_text`, `WandbLogger.log_image` and `WandbLogger.log_table` |

```Python
from pytorch_lightning.loggers import WandbLogger
# define Wandb Logger
logger = WandbLogger(
    name=hparams['run'],
    project=hparams['project_name'],
    version=hparams['run'],
    save_dir='kaggle/working/checkpoints'
)
```

#### [Optinal] Defining Custom Callbacks

##### Using `WandbLogger` to log Images, Text and More

- Pytorch Lightning is extensible through its callback system to create custom callback to automatically log sample predictions during validation.
- `WandbLogger` provides convenient media logging functions:
  - `WandbLogger.log_text` for text data
  - `WandbLogger.log_image` for images
  - `WandbLogger.log_table` for [W&B Tables](https://docs.wandb.ai/guides/data-vis).
- For example, we want to log the first 20 images in the first batch of the validation dataset along with the predicted and ground truth labels.

```Python
from pytorch_lightning.callbacks import Callback

class LogPredictionsCallback(Callback):

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]

            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger.log_image(key='sample_images', images=images, caption=captions)

            # Option 2: log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            wandb_logger.log_table(key='sample_table', columns=columns, data=data)

log_predictions_callback = LogPredictionsCallback()
```

### Defining `Trainer`

- `benchmark=True` PyTorch Lightning will attempt to find the fastest implementation of operations for the current hardware during the first training epoch, potentially improving training performance.
- `devices=1`: Indicates the number of devices to be used. In this case, you've specified 1, which means training will be done on a single GPU.
- `gradient_clip_val=10.`: Specifies the maximum norm of gradients. Gradients exceeding this value will be clipped during optimization, which can help prevent exploding gradients.

```Python
trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            strategy='ddp_notebook',
            logger=logger, # provide Wandb logger
            max_epochs=hparams['epochs'],
            benchmark=True,
            callbacks=[ckpt_callback, log_predictions_callback], # provide callbacks
            gradient_clip_val=10.,
            enable_model_summary=True,
)
# to perform the training & hyper-parameter tuning
trainer.fit(model)
# to eval the model
trainer.test(model)
```

- When we want to close our W&B run, we call `wandb.finish()` (mainly useful in notebooks, called automatically in scripts).

```Python
wandb.finish()
```

### Loading Model from Checkpoints

```Python
trained_model = VAE.load_from_checkpoint('./model_checkpoints/vae_weights.ckpt')
trained_model.freeze() # for inference
```
