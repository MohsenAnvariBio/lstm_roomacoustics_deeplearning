import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Read impulse response WAV file
# --------------------------------------------------
def read_rir_wav(filepath):
    rir, fs = sf.read(filepath)
    if rir.ndim > 1:
        rir = rir[:, 0]  # convert to mono if stereo
    return rir, fs


# --------------------------------------------------
# 2. Estimate effective bandwidth (energy-based)
# --------------------------------------------------
def estimate_bandwidth(signal, fs, energy_ratio=0.99):
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1 / fs)

    power = np.abs(fft_vals) ** 2
    cumulative_energy = np.cumsum(power)
    cumulative_energy /= cumulative_energy[-1]

    idx = np.where(cumulative_energy >= energy_ratio)[0][0]
    f_max = freqs[idx]

    return f_max


# --------------------------------------------------
# 3. Automatically create frequency bands
# --------------------------------------------------
def create_frequency_bands(f_min, f_max, num_bands=10):
    edges = np.linspace(f_min, f_max, num_bands + 1)
    return [(edges[i], edges[i + 1]) for i in range(num_bands)]


# --------------------------------------------------
# 4. Bandpass filter
# --------------------------------------------------
def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# --------------------------------------------------
# 5. Energy Decay Curve (Schroeder integration, dB)
# --------------------------------------------------
def compute_edc_db(signal):
    energy = signal ** 2
    edc = np.flip(np.cumsum(np.flip(energy)))
    edc /= np.max(edc) + 1e-12
    edc_db = 10 * np.log10(edc + 1e-12)
    return edc_db


# --------------------------------------------------
# 6. Fix time length (pad or truncate)
# --------------------------------------------------
def fix_length(matrix, target_length):
    """
    matrix shape: (time, bands)
    """
    time, bands = matrix.shape

    if time > target_length:
        return matrix[:target_length, :]
    else:
        pad_len = target_length - time
        return np.pad(matrix, ((0, pad_len), (0, 0)))


# --------------------------------------------------
# 7. Full automatic RIR → EDC matrix pipeline
# --------------------------------------------------
def rir_to_edc_matrix_auto(rir, fs, num_bands=10, f_min=20):
    f_max = estimate_bandwidth(rir, fs)
    bands = create_frequency_bands(f_min, f_max, num_bands)

    edc_list = []
    for low, high in bands:
        subband = bandpass_filter(rir, fs, low, high)
        edc_db = compute_edc_db(subband)
        edc_list.append(edc_db)

    # (time, bands)
    edc_matrix = np.stack(edc_list, axis=1)

    return edc_matrix, bands


# --------------------------------------------------
# 8. Process all RIR WAV files (FINAL DATASET)
# --------------------------------------------------
def process_rir_folder(folder_path, num_bands=10, target_length=144000):
    edc_outputs = []
    band_info = []

    wav_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    )

    for file in wav_files:
        filepath = os.path.join(folder_path, file)
        rir, fs = read_rir_wav(filepath)

        edc_matrix, bands = rir_to_edc_matrix_auto(
            rir, fs, num_bands=num_bands
        )

        # Fix time length
        edc_matrix = fix_length(edc_matrix, target_length)

        edc_outputs.append(edc_matrix)
        band_info.append(bands)

    # Convert to NumPy array → (num_wavs, time, bands)
    edc_outputs = np.array(edc_outputs)

    return edc_outputs, band_info, fs


# --------------------------------------------------
# 9. Plot ONE RIR (optional sanity check)
# --------------------------------------------------
def plot_edc_one_rir(edc_matrix, fs, title="EDC (dB) per Frequency Band"):
    time_samples, num_bands = edc_matrix.shape
    time_axis = np.arange(time_samples) / fs

    plt.figure(figsize=(10, 6))

    for band_idx in range(num_bands):
        plt.plot(time_axis, edc_matrix[:, band_idx],
                 label=f"Band {band_idx + 1}")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Energy Decay (dB)")
    plt.title(title)
    plt.ylim(0, -60)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 10. Main
# --------------------------------------------------
if __name__ == "__main__":
    rir_folder = "RIR"

    # Build dataset
    edc_data, bands_used, fs = process_rir_folder(
        rir_folder,
        num_bands=10,
        target_length=144000
    )

    print("Final dataset shape:", edc_data.shape)

    # Save dataset (Option 1: NumPy)
    np.save("edc_dataset.npy", edc_data)
    print("Saved: edc_dataset.npy")

    # Plot first RIR for verification
    plot_edc_one_rir(
        edc_data[0],
        fs,
        title="EDC of First RIR (10 Bands, dB)"
    )
