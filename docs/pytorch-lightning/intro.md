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

### `Dataset` Class Preparation

- Prepare the dataset class that is able to load either `train` or `test` data via `split` variable

```Python
from torch.utils.data import Dataset

class TsDataset(Dataset):
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
        return torch.tensor(self.cont[idx]), torch.tensor(self.cat[idx]), torch.tensor(self.target[idx])

    def __len__(self):
        return self.df.shape[0]
```
