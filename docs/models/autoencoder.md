# Autoencoder

## Introduction

### What is the autoencoder ?

- An autoencoder is a type of unsupervised neural network architecture designed to efficiently compress (**encode**) input data down to its essential features, then reconstruct (**decode**) the original input from this compressed representation.
- Autoencoders are trained to discover latent variables of the input data: hidden or random variables that, despite not being directly observable, fundamentally inform the way data is distributed.
  - The latent variables of a given set of input data are referred to as **latent space**.

### Autoencoder applications

- Most types of autoencoders are used for
  - Feature extraction tasks: data compression, image denoising, anomaly detection and facial recognition.
  - Generative tasks: image generation or generating time series data.

### Autoencoders vs Encoder-Decoders

- Though all autoencoder models include both an encoder and a decoder, not all encoder-decoder models are autoencoders.
- In most applications of encoder-decoder models, the output of the neural network is different from its input.
- Autoencoders refer to a specific subset of encoder-decoder architectures that are trained via unsupervised learning to reconstruct their own input data, so the output is the same as the input for autoencoders

## Vanilla Autoencoders

- _Summary - Vanilla Autoencoders_: encoder + latent vectors (compressed representation) + decoder
- In practice, such classical autoencoders don’t lead to particularly useful or nicely structured latent spaces. They’re not much good at compression, either.

## Variational Autoencoders

- _Summary - Variational Autoencoder_: encoder + latent normal distribution defined by `z_mean` and `z_log_var` + sample + decoder
  - [Architecture Diagram](https://miro.medium.com/v2/resize:fit:720/format:webp/1*3ixSeYAGPvsPtndYconxMg.png)
- Variational Autoencoders (VAEs) are generative models that learn compressed representations of their training data as probability distributions, which are used to generate new sample data by creating variations of those learned representations.
- The fundamental difference between VAEs and other types of autoencoders is that while most autoencoders learn _discrete_ latent space models, VAEs learn _continuous_ latent variable models.
- VAEs model two different vectors:
  - a vector of means, $μ$ or `z_mean`
  - a vector of standard deviations, $σ$ or `z_log_var`
  - Because these vectors capture latent attributes as a probability distribution — that is, they learn a _stochastic_ encoding rather than a _deterministic_ encoding
- To generate a new sample `z`, the VAE samples a random latent vector ($ε$ or `epsilon`) from within the unit Gaussian (in other words, selects a random starting point from within the normal distribution) and shifts it by the mean of the latent distribution ($μ$ or `z_mean`) and scales it by the variance of the latent distribution ($σ$ or `z_log_var`).

  - Formula: `z = z_mean + exp(0.5 * z_log_variance) * epsilon` where `epsilon` is a random tensor of small values (sampled from a standard normal distribution)
  - This process, called the **reparameterization** trick

  ```Python
  def reparameterize(self, mu, logvar):
    '''
    The reparameterisation trick allows us to backpropagate through the encoder.
    This allows the mean and log-variance vectors to still remain
    as the learnable parameters of the network while still maintaining the stochasticity of the entire system via epsilon.
    '''
    if self.training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * self.hparams.stdev
        return eps * std + mu
    else:
        return mu

  ```

- Loss functions: the parameters of a VAE are trained via

  - **Reconstruction Loss** that forces the decoded samples to match the initial inputs:
    - _L2 loss_ or _Binary Crossentropy_ or Huber loss with $𝛿=1$ (called `smooth_l1_loss` in Pytorch)
    - Note: Mean Square Error (MSE) is quite sensitive to outliers, so MSE is not selected
  - **Regularization Loss** that nudges the distribution of the encoder output toward a well-rounded normal distribution centered around 0. This provides the encoder with a sensible assumption about the structure of the latent space it’s modeling: _Kullback-Liebler Divergence_. An excellent interpretation of KL Divergence is available in GANs in Action (by Jakub Langr and Vladimir Bok) (Page 29, 1st Edition)
  - Total Loss = Reconstruction Loss + kld_beta \* Regularization Loss
    - `kld_beta`, the coefficient applied to the KLD loss term in the total loss computation. This helps because the reconstruction loss is typically harder to improve than the KLD loss, therefore if both were weighted equally, the model would start by optimising the KLD loss before improving the reconstruction loss substantially. This hyperparameter means that our VAE is technically a $\beta-\text {VAE}$.

  ```Python
  def loss_function(self, obs, recon, mu, logvar):
    recon_loss = F.smooth_l1_loss(recon, obs, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
    return recon_loss, kld
  ```

- Example of VAE's encoder

```Python
# define the hyper-parameters of the model
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

# encoder module
class Encoder(nn.Module):
    '''
    The encoder part of our VAE. Takes a data sample and returns the mean and the log-variance of the
    latent vector's distribution.
    '''
    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)
        self.embeds = nn.ModuleList([
            # create embedding for each cat feature: nn.Embedding(n_cats, emb_size), where
            # n_cats: number of unique value in the cat feature
            # emb_size: the size of each embedding vector
            nn.Embedding(n_cats, emb_size) for (n_cats, emb_size) in self.hparams.embedding_sizes
        ])
        # The input to the first layer is the concatenation of all embedding vectors and continuous
        # values
        in_dim = sum(emb.embedding_dim for emb in self.embeds) + len(self.hparams.cont_vars)
        layer_dims = [in_dim] + [int(s) for s in self.hparams.layer_sizes.split(',')] # in_dim + '64,128,256,128,64'
        bn = self.hparams.batch_norm
        self.layers = nn.Sequential(
            *[Layer(layer_dims[i], layer_dims[i + 1], bn) for i in range(len(layer_dims) - 1)],
        )
        # latent mean (mu) and log_variance (logvar)
        self.mu = nn.Linear(layer_dims[-1], self.hparams.latent_dim)     # latent_dim=16: latent vector dimension
        self.logvar = nn.Linear(layer_dims[-1], self.hparams.latent_dim)

    def forward(self, x_cont, x_cat):
        x_embed = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x_embed = torch.cat(x_embed, dim=1) # concat all emb_vectors into a single vector x_embed
        x = torch.cat((x_embed, x_cont), dim=1) # concat x_embed with cont vector
        h = self.layers(x)
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        return mu_, logvar_, x  # we return the concatenated input vector for use in loss fn
```

- Example of VAE's decoder

```Python
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
```

- Example of VAE with Pytorch Lightning module

```Python
class VAE(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)

    def reparameterize(self, mu, logvar):
        '''
        The reparameterisation trick allows us to backpropagate through the encoder.
        This allows the mean and log-variance vectors to still remain
        as the learnable parameters of the network while still maintaining the stochasticity of the entire system via epsilon.
        '''
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * self.hparams.stdev
            return eps * std + mu
        else:
            return mu

    def forward(self, batch):
        x_cont, x_cat = batch
        assert x_cat.dtype == torch.int64
        mu, logvar, x = self.encoder(x_cont, x_cat)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, x

    def loss_function(self, obs, recon, mu, logvar):
        recon_loss = F.smooth_l1_loss(recon, obs, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        return recon_loss, kld

    def training_step(self, batch, batch_idx):
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
        recon, mu, logvar, x = self.forward(batch)
        recon_loss, kld = self.loss_function(x, recon, mu, logvar)
        loss = recon_loss + self.hparams.kld_beta * kld
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay,
                                eps=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=25, T_mult=1, eta_min=1e-9, last_epoch=-1)
        return [opt], [sch]

    def train_dataloader(self):
        dataset = TsDataset('train', data_folder_path=self.hparams.data_folder_path, cont_vars=self.hparams.cont_vars,
            cat_vars = self.hparams.cat_vars, lbl_as_feat=True
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=2,
            pin_memory=True, persistent_workers=True, shuffle=True
        )

    def test_dataloader(self):
        dataset = TsDataset('test', data_folder_path=self.hparams.data_folder_path, cont_vars=self.hparams.cont_vars,
            cat_vars=self.hparams.cat_vars, lbl_as_feat=True
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=2,
            pin_memory=True, persistent_workers=True
        )

model = VAE(**hparams)
```
