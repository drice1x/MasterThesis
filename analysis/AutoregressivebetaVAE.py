import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

# Define the BetaVAE architecture
class BetaVAE(nn.Module):
    def __init__(self, observation_dim, latent_dim, beta):
        super(BetaVAE, self).__init__()
        self.beta = beta

        # Encoder for observation
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # Twice the latent_dim for mean and log variance
        )

        # Decoder for observation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, observation_dim),
            nn.Sigmoid()
        )
            # Set the default data type of model parameters to float64
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.double()
                if module.bias is not None:
                    module.bias.data = module.bias.data.double()

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)  # Split the output into mu and logvariance
        return mu, logvar


    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encode(x)
        #mean, logvar = encoded[:, :latent_dim], encoded[:, latent_dim:]
        z = self.reparameterize(mean, logvar)
        decoded = self.decode(z)
        return decoded, mean, logvar

# Load and preprocess data
hdf5_file = h5py.File('/home/paperspace/decision-diffuser/code/analysis/my_dataset.hdf5', 'r')
dataset_name = 'observations'
observations = hdf5_file[dataset_name][()]
dataset_name1 = 'terminals'
terminals = hdf5_file[dataset_name1][()]
df_obs = pd.DataFrame(observations)
obs = df_obs.values

hdf5_file.close()

# Create a dataset
class MyDataset(Dataset):
    def __init__(self, observations, terminals):
        self.observations = observations
        self.terminals = terminals

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = self.observations[idx]
        is_terminal = self.terminals[idx]

        return observation, is_terminal

# Create a dataset with variable-length trajectories
dataset = MyDataset(obs, terminals)

# Create a dataloader for batching
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters
observation_dim = obs.shape[1]
latent_dim = 12
beta = 0.5
num_epochs = 20
learning_rate = 1e-3

# Initialize β-VAE
vae = BetaVAE(observation_dim, latent_dim, beta)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Inside the training loop
for epoch in range(num_epochs):
    for batch, is_terminal in dataloader:
        optimizer.zero_grad()
        observations = batch

        # Only use the first observation for autoregressive decoding
        observation = torch.tensor(observations[0]).unsqueeze(0)

        # Initialize the latent state using the first observation
        #encoded = vae.encode(observation)
        #mean, logvar = encoded[:, :latent_dim], encoded[:, latent_dim:]
        mean, logvar = vae.encode(observation)
        z = vae.reparameterize(mean, logvar)

        # Autoregressive decoding
        generated_observations = []
        for t in range(1, len(observations)):
            # Use the autoregressive decoder to generate the next observation
            autoregressive_input = torch.cat((z, observations[t-1].unsqueeze(0)), dim=1)
            autoregressive_input1 = torch.cat((z, observations[t-1].unsqueeze(0).repeat(z.shape[0], 1)), dim=1)
            generated_observation = vae.decode(autoregressive_input1)
            generated_observations.append(generated_observation)

            # Update the latent state using the generated observation
            #encoded = vae.encode(generated_observation)
            #mean, logvar = encoded[:, :latent_dim], encoded[:, latent_dim:]
            mean, logvar = vae.encode(generated_observation)
            z = vae.reparameterize(mean, logvar)

        # Compute the autoregressive reconstruction loss (MSE) for each time step
        autoregressive_loss = sum([nn.MSELoss()(gen_observation, observations[t]) for t, gen_observation in enumerate(generated_observations)])

        # Compute KL divergence term
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Combine autoregressive loss, KL divergence, and beta
        loss = autoregressive_loss + beta * kl_divergence

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Handle the ends of trajectories
        if any(is_terminal):
            # Reset the latent state when a trajectory ends
            encoded = vae.encode(torch.cat((observation, generated_observations[-1]), dim=1))
            mean, logvar = encoded[:, :latent_dim], encoded[:, latent_dim:]
            z = vae.reparameterize(mean, logvar)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")