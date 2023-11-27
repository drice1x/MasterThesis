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
            nn.Linear(observation_dim, 30),
            nn.ReLU(),
            nn.Linear(30, latent_dim * 2)  # Twice the latent_dim for mean and log variance
        )

        # Decoder for observation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 30),
            nn.ReLU(),
            nn.Linear(30, observation_dim),
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
        z = self.reparameterize(mean, logvar)
        decoded = self.decode(z)
        return decoded, mean, logvar

def loadtrain ():
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
        def __init__(self, observations):
            self.observations = observations

        def __len__(self):
            return len(self.observations)

        def __getitem__(self, idx):
            observation = self.observations[idx]
            return observation

    # Create a dataset with variable-length trajectories
    dataset = MyDataset(obs)

    # Create a dataloader for batching
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Hyperparameters
    observation_dim = obs.shape[1]
    latent_dim = 12
    beta = 0.2
    num_epochs = 20
    learning_rate = 1e-3

    # Initialize β-VAE
    vae = BetaVAE(observation_dim, latent_dim, beta)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # Additional hyperparameter for stopping loss
    stopping_loss_weight = 0.1  # Adjust this weight as needed

    # Inside the training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            observations = batch

            # Forward pass
            reconstructed_observations, mean, logvar = vae(observations)

            # Compute reconstruction loss (MSE) and KL divergence term
            reconstruction_loss = nn.MSELoss()(reconstructed_observations, observations)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

            # Combine reconstruction loss, KL divergence, and beta
            loss = reconstruction_loss + beta * kl_divergence

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
    # After training
    # Save the trained autoencoder model
    model_filename = 'vae0709.pth'
    model_path = '/home/paperspace/decision-diffuser/code/vae/' + model_filename
    torch.save(vae.state_dict(), model_path)

    print("stop")

    # Create a new instance of the BetaVAE model with the same architecture and hyperparameters
    #loaded_vae = BetaVAE(observation_dim, latent_dim, beta)

    # Load the trained model state
    #loaded_vae.load_state_dict(torch.load('trained_vae_model.pth'))