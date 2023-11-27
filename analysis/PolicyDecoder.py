import torch
import torch.nn as nn
import torch.optim as optim
import code.analysis.AutoregressivebetaVAE as AutoregressivebetaVAE



# Policy Decoder Network
class PolicyDecoder(nn.Module):
    def __init__(self, trajectory_dim, latent_dim, action_dim):
        super(PolicyDecoder, self).__init__()
        self.trajectory_dim = trajectory_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        self.trajectory_encoder = nn.Linear(trajectory_dim, latent_dim)
        self.state_encoder = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, encoded_trajectory, encoded_state):
        encoded_trajectory = self.trajectory_encoder(encoded_trajectory)
        encoded_state = self.state_encoder(encoded_state)
        combined = torch.cat((encoded_trajectory, encoded_state), dim=1)
        action_prediction = self.decoder(combined)
        return action_prediction

# Example usage
trajectory_dim = 16  # Dimension of encoded trajectory from β-VAE
latent_dim = 16  # Dimension of encoded state for time step T+1
action_dim = 2  # Dimension of action

# Initialize your β-VAE
beta_vae = AutoregressivebetaVAE(observation_dim, action_dim, latent_dim, beta)
# Initialize your trajectory encoder for state at T+1 (modify according to your design)
trajectory_encoder = nn.Linear(observation_dim, latent_dim)
# Assuming you have your dataset and dataloader set up for training
# ...

policy_decoder = PolicyDecoder(trajectory_dim, latent_dim, action_dim)

# Load and preprocess your dataset for training
# ...

# Assuming dataloader provides N+1 encoded observations and separately encoded state for each training example
criterion = nn.MSELoss()
optimizer = optim.Adam(policy_decoder.parameters(), lr=0.001)

# Training loop for Policy Decoder
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        encoded_trajectory_N = batch[0]  # Encoded trajectory from β-VAE up to time step T
        encoded_state_T_plus_1 = batch[1]  # Encoded state for time step T+1
        target_action_T_plus_1 = batch[2]  # Target action for time step T+1
        
        # Decode the N encoded observations for the trajectory up to time step T
        decoded_trajectory_N, _, _ = beta_vae.decode(encoded_trajectory_N)
        
        # Pass the decoded trajectory up to time step T and encoded state at T+1 through the policy decoder
        predicted_action_T_plus_1 = policy_decoder(decoded_trajectory_N, encoded_state_T_plus_1)
        
        loss = criterion(predicted_action_T_plus_1, target_action_T_plus_1)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
