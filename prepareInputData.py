import pandas as pd
import numpy as np

CSV_PATH = "roomFeaturesDataset.csv"
OUTPUT_NPY = "input_features.npy"

df = pd.read_csv(CSV_PATH)

print("Original CSV shape:", df.shape)
print("Columns:", df.columns.tolist())

feature_columns = [
    "Length", "Width", "Height",
    "Source_X", "Source_Y", "Source_Z",
    "Receiver_X", "Receiver_Y", "Receiver_Z",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7"
]

X = df[feature_columns].values.astype(np.float32)
assert X.shape[1] == 16, "Input feature dimension must be 16"
print("Final input feature shape:", X.shape)

# Save
np.save(OUTPUT_NPY, X)
print(f"Saved input features to {OUTPUT_NPY}")
