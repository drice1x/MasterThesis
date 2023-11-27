import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt

size = {
    1: (16, 8,  4),   
    2: (36, 24, 16),
    3: (64, 48, 36)
}

sizes = size[2]

class Autoencoder(nn.Module):
    def __init__(self, input_dim=sizes[0], latent_dim=sizes[2]):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, sizes[1]),
            nn.ReLU(),
            nn.Linear(sizes[1], latent_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, sizes[1]),
            nn.ReLU(),
            nn.Linear(sizes[1], input_dim),
            nn.Tanh(),
            #nn.Linear(latent_dim, input_dim),
            #nn.Sigmoid(),
        )
        self.min_val = None
        self.max_val = None

        # Set the default data type of model parameters to float64
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.double()
                if module.bias is not None:
                    module.bias.data = module.bias.data.double()

    def fit_normalization(self, data):
        self.min_val = data.min()
        self.max_val = data.max()

    def normalize_data(self, data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalization parameters are not set. Call fit_normalization(data) first.")
        return (data - self.min_val) / (self.max_val - self.min_val)

    def encode(self, x):
        #normalized_x = self.normalize_data(x)
        return self.encoder(x) #self.encoder(normalized_x) #self.encoder(x) #

    def decode(self, z):
        decoded_z = self.decoder(z)
        return decoded_z
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



def load_and_train():

    # Load and preprocess data
    hdf5_file = h5py.File('/home/paperspace/decision-diffuser/code/analysis/newdataset/withoutGoal/my_datasetExperiment1.hdf5', 'r')
    dataset_name = 'observations'
    data = hdf5_file[dataset_name][()]
    df = pd.DataFrame(data)
    Data = df.values
    hdf5_file.close()

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split

    data, X_test = train_test_split(Data, test_size=0.15, random_state=42)



    # Create autoencoder model and move it to GPU if available
    device = torch.device("cuda" )#if torch.cuda.is_available() else "cpu")
    autoencoder = Autoencoder().to(device)

    # Fit normalization
    #autoencoder.fit_normalization(data)

    # Convert data to PyTorch tensor and move to GPU
    #normalized_input_data = torch.tensor(autoencoder.normalize_data(data), dtype=torch.double).to(device)
    normalized_input_data = torch.tensor(data, dtype=torch.double).to(device)
    # Assuming you've loaded your data and processed it into X_train
    X_train = torch.tensor(normalized_input_data, dtype=torch.double).to(device)

    # Create an instance of the Autoencoder class
    #autoencoder = Autoencoder(input_dim=64, latent_dim=16).to(device)




    # Create an instance of the Autoencoder class
    autoencoder = Autoencoder(input_dim=sizes[0], latent_dim=sizes[2]).to(device)
    # Define your loss criterion and optimizer
    criterion = nn.MSELoss()
    weight_decay = 1e-6  # Adjust the weight decay value as needed
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001, weight_decay=weight_decay)

    # Set stopping criterion parameters
    prev_avg_loss = float('inf')  # Initialize with a high value
    loss_threshold = 1e-5  # Adjust the threshold as needed

    # Train the autoencoder
    num_epochs = 150
    batch_size = 32
    # Lists to store training loss and validation loss for plotting
    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        autoencoder.train()
        total_loss = 0.0

        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size].to(device)
            optimizer.zero_grad()
            outputs = autoencoder(batch)
            loss = criterion(outputs, batch)
            
            # Compute L2 regularization term
            l2_reg = torch.tensor(0.0, dtype=torch.double).to(device)
            for param in autoencoder.parameters():
                l2_reg += torch.norm(param)
            loss += weight_decay * l2_reg
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (len(X_train) // batch_size)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg. Loss: {avg_loss:.4f}")

        # Append training loss for plotting
        train_losses.append(avg_loss)

        # Check stopping criterion
        loss_change = prev_avg_loss - avg_loss
        if abs(loss_change) < loss_threshold:
            print("Stopping criterion met. Stopping training.")
            break

        prev_avg_loss = avg_loss

        # Validation step
        if (epoch + 1) % 10 == 0:  # Run validation every 10 epochs
            autoencoder.eval()
            with torch.no_grad():
                val_outputs = autoencoder(X_train.to(device))
                val_loss = criterion(val_outputs, X_train.to(device)).item()
            print(f"Validation Loss: {val_loss:.4f}")
            # Append validation loss for plotting
            validation_losses.append(val_loss)
    # Save the trained autoencoder model
    model_filename = 'autoencoderEnv1.pth'
    model_path = '/home/paperspace/decision-diffuser/code/autoencoder/newae/' + model_filename

    torch.save(autoencoder.state_dict(), model_path)

    # %%
    '''# Plot training and validation loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(10, num_epochs + 1, 10), validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()'''
    # %%
    print("now begins the evaluation:")

    # Convert testing data to PyTorch tensor and move to GPU if available
    X_test = torch.tensor(X_test, dtype=torch.double).to(device)  # Use the same device you used for training

    # Evaluate the trained autoencoder on the testing data
    autoencoder.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_outputs = autoencoder(X_test)
        test_loss = criterion(test_outputs, X_test).item()
    print(f"Test Loss: {test_loss:.4f}")


    print("now begins the visualization of samples")

    #import matplotlib.pyplot as plt
    # %%
    num_samples_to_visualize = 3
    samples = X_test[:num_samples_to_visualize].cpu().numpy()
    reconstructed_samples = test_outputs[:num_samples_to_visualize].cpu().numpy()

    print("hallo:", type(samples))
    print(type(reconstructed_samples))

    # %%
    import pandas as pd

    # Create a DataFrame to store the loss lists
    loss_df = pd.DataFrame({'Epoch': range(1, len(train_losses)+1), 'Training Loss': train_losses})

    # Create a DataFrame to store the loss lists
    val_df = pd.DataFrame({'Epoch': range(1, len(validation_losses)+1), 'Validation Loss': validation_losses})


    # Save the loss data to a CSV file
    loss_df.to_csv('/home/paperspace/decision-diffuser/code/analysis/DATAANALYSIS/Trainingloss_dataAE3_100t.csv', index=False)
    val_df.to_csv('/home/paperspace/decision-diffuser/code/analysis/DATAANALYSIS/Valloss_dataAE3_100t.csv', index=False)
    # Create a DataFrame to store the samples and their reconstructions
    sampless= samples[:num_samples_to_visualize]

    recon = reconstructed_samples[:num_samples_to_visualize]
    sampless = []
    recon = []

    for i in range(num_samples_to_visualize):
        sampless.append(samples[i].flatten())
        recon.append(reconstructed_samples[i].flatten())

    # Create the DataFrame
    samples_df = pd.DataFrame({'Sample Index': range(num_samples_to_visualize), "Original Sample": sampless, "Reconstructed Sample": recon})

    # Save the DataFrame to a CSV file
    samples_df.to_csv('/home/paperspace/decision-diffuser/code/analysis/DATAANALYSIS/samples_dataAE3_100t.csv', index=False)

    print("for debug last row")
