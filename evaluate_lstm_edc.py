import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# =========================
# Configuration (MUST match training)
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIME_STEPS = 144000
NUM_BANDS = 10
INPUT_DIM = 16
LATENT_DIM = 128
LSTM_LAYERS = 2

# =========================
# Model definition (same as training)
# =========================
class LSTM_EDC_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_DIM),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=LATENT_DIM,
            hidden_size=LATENT_DIM,
            num_layers=LSTM_LAYERS,
            batch_first=True
        )

        self.output_layer = nn.Linear(LATENT_DIM, NUM_BANDS)

    def forward(self, x):
        batch_size = x.size(0)

        latent = self.encoder(x)
        latent = latent.unsqueeze(1)
        latent = latent.repeat(1, TIME_STEPS, 1)

        lstm_out, _ = self.lstm(latent)
        out = self.output_layer(lstm_out)

        return out

# =========================
# Load model and scalers
# =========================
model = LSTM_EDC_Model().to(DEVICE)
model.load_state_dict(torch.load("lstm_edc_model.pth", map_location=DEVICE))
model.eval()

x_scaler = joblib.load("x_scaler.save")
y_scalers = joblib.load("y_scalers.save")

# =========================
# Load data
# =========================
X = np.load("input_features.npy")
Y = np.load("edc_dataset.npy")

# Scale input
X_scaled = x_scaler.transform(X)

# =========================
# Select ONE room
# =========================
sample_idx = 0

X_sample = torch.tensor(
    X_scaled[sample_idx:sample_idx+1],
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():
    Y_pred_scaled = model(X_sample).cpu().numpy()

# =========================
# Inverse normalization
# =========================
Y_true = Y[sample_idx]
Y_pred = np.zeros_like(Y_true)

for b in range(NUM_BANDS):
    Y_pred[:, b] = y_scalers[b].inverse_transform(
        Y_pred_scaled[0, :, b].reshape(-1, 1)
    ).ravel()

# =========================
# Plot GT vs Prediction
# =========================
time_axis = np.arange(TIME_STEPS)

plt.figure(figsize=(12, 8))

for b in range(NUM_BANDS):
    plt.subplot(5, 2, b + 1)
    plt.plot(time_axis, Y_true[:, b], label="Ground Truth")
    plt.plot(time_axis, Y_pred[:, b], "--", label="Prediction")
    plt.title(f"Band {b+1}")
    plt.xlabel("Time samples")
    plt.ylabel("EDC (dB)")
    plt.grid(True)

plt.legend()
plt.tight_layout()
plt.show()
