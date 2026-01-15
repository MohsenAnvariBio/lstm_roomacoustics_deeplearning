import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os

# =========================
# Configuration
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2           # keep small (long sequences!)
EPOCHS = 3 #--------------50
LEARNING_RATE = 1e-3

TIME_STEPS = 144000

NUM_BANDS = 10
INPUT_DIM = 16
LATENT_DIM = 128
LSTM_LAYERS = 2

# =========================
# Dataset
# =========================
class EDCDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# =========================
# Model
# =========================
class LSTM_EDC_Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (static → latent)
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_DIM),
            nn.ReLU()
        )

        # LSTM Decoder
        self.lstm = nn.LSTM(
            input_size=LATENT_DIM,
            hidden_size=LATENT_DIM,
            num_layers=LSTM_LAYERS,
            batch_first=True
        )

        # Output layer (per time-step)
        self.output_layer = nn.Linear(LATENT_DIM, NUM_BANDS)

    def forward(self, x):
        """
        x shape: (batch, 16)
        output: (batch, 144000, 10)
        """
        batch_size = x.size(0)

        latent = self.encoder(x)                     # (batch, 128)
        latent = latent.unsqueeze(1)                 # (batch, 1, 128)
        latent = latent.repeat(1, TIME_STEPS, 1)     # (batch, 144000, 128)

        lstm_out, _ = self.lstm(latent)               # (batch, 144000, 128)
        out = self.output_layer(lstm_out)             # (batch, 144000, 10)

        return out

# =========================
# Load Data
# =========================
X = np.load("input_features.npy")     # (n, 16)
#X = X[:20] 
Y = np.load("edc_dataset.npy")        # (n, 144000, 10)

print("Input shape:", X.shape)
print("Output shape:", Y.shape)

# =========================
# Normalization (Professor-style)
# =========================
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

# Normalize EDC per band
y_scalers = []
Y_scaled = np.zeros_like(Y)

for b in range(NUM_BANDS):
    scaler = StandardScaler()
    Y_scaled[:, :, b] = scaler.fit_transform(Y[:, :, b])
    y_scalers.append(scaler)

# Save scalers
joblib.dump(x_scaler, "x_scaler.save")
joblib.dump(y_scalers, "y_scalers.save")

# =========================
# Dataset & Loader
# =========================
dataset = EDCDataset(X_scaled, Y_scaled)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

# =========================
# Training Setup
# =========================
model = LSTM_EDC_Model().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# Training Loop
# =========================
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for X_batch, Y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.6f}")

# =========================
# Save Model
# =========================
torch.save(model.state_dict(), "lstm_edc_model.pth")
print("Training complete. Model saved.")
