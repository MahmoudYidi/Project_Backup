import sys
import os
from network_utils import *


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



# Initialisations
start_wavelength = 529.91
end_wavelength = 580.80
filepath = "/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/data_paths.txt"
epochs = 5
batch_size = 1
height = 800
width = 1024
num_bands = 39


# Define VAE architecture
# Define VAE architecture
# Define hyperparameters
height, width, bands = 800, 1024, 39  # Define the input shape
latent_dim = 32
epochs = 100
batch_size = 1

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0], out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * (input_dim[1] // 4) * (input_dim[2] // 4), latent_dim)
        self.fc_logvar = nn.Linear(64 * (input_dim[1] // 4) * (input_dim[2] // 4), latent_dim)

        # Decoder layers
        self.decoder_input = nn.Linear(latent_dim, 64 * (input_dim[1] // 4) * (input_dim[2] // 4))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=input_dim[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.decoder_input(z)
        x = x.view(-1, 64, self.input_dim[1] // 4, self.input_dim[2] // 4)
        reconstruction = self.decoder(x)
        return reconstruction, mu, logvar

# Define the loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, input_dim[0] * input_dim[1] * input_dim[2]),
                                             x.view(-1, input_dim[0] * input_dim[1] * input_dim[2]), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Load your dataset
def load_data():
        #Data Loading
    hsi_data_list, selected_wavelengths_list = load_envi_hsi_by_wavelengths_net(filepath, start_wavelength, end_wavelength)

    #Check Data
    for hsi_data, selected_wavelengths in zip(hsi_data_list, selected_wavelengths_list):
        print("Loaded HSI data shape:", hsi_data.shape)
        
    preprocessed_data_list = []
    for hsi_data in hsi_data_list:
        # Normalize intensity values to range [0, 1]
        hsi_data_normalized = hsi_data / np.max(hsi_data)
        # Optionally, reshape the data if needed
        # hsi_data_reshaped = hsi_data_normalized.reshape(...)

        # Append preprocessed data to list
        preprocessed_data_list.append(hsi_data_normalized)

    # Convert list to numpy array
    preprocessed_data_array = np.array(preprocessed_data_list)

    # Print shape of preprocessed data
    print("Preprocessed data shape:", preprocessed_data_array.shape)
    hsi_tensor = torch.tensor(preprocessed_data_array, dtype=torch.float32)
    return hsi_tensor
    #pass


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the data
data = load_data()

# Transpose the data to have channels as the second dimension
data = data.permute(0, 3, 1, 2)

# Move data to the device
data = data.to(device)

# Update input dimension
input_dim = (bands, height, width)

# Create DataLoader
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)


# Initialize the VAE model
model = VAE(input_dim, latent_dim).to(device)
print("intialised!!!")
# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = vae_loss(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, total_loss / len(dataloader.dataset)))
torch.save(model.state_dict(), 'vae_model.pth')
