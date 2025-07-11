import numpy as np
import spherical_harmonics as sh
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def analyze_and_visualize_ir(ir, sample_rate, distances, speeds, frequency_points):
    """
    Comprehensive analysis and visualization of ambisonic IR data
    """
    # Create a figure with multiple subplots using GridSpec
    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(4, 2, figure=fig)
    
    # Waveform visualization for first few channels
    ax1 = fig.add_subplot(gs[0, :])
    time_axis = np.arange(ir.shape[1]) / sample_rate * 1000  # Convert to ms
    for i in range(min(4, ir.shape[0])):
        ax1.plot(time_axis, ir[i], label=f'Channel {i}', alpha=0.7)
    ax1.set_title('Ambisonic IR Waveforms (First 4 Channels)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True)
    
    # Energy decay curve
    ax2 = fig.add_subplot(gs[1, 0])
    edc = np.cumsum(np.sum(ir**2, axis=0)[::-1])[::-1]
    edc_db = 10 * np.log10(edc / np.max(edc) + 1e-10)
    ax2.plot(time_axis, edc_db)
    ax2.set_title('Energy Decay Curve')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Energy (dB)')
    ax2.grid(True)
    ax2.set_ylim([-60, 0])
    
    # Spectrogram of channel 0
    ax3 = fig.add_subplot(gs[1, 1])
    nperseg = min(1024, ir.shape[1]//4)
    ax3.specgram(ir[0], Fs=sample_rate, NFFT=nperseg, noverlap=nperseg//2,
                cmap='viridis')
    ax3.set_title('Spectrogram (Channel 0)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    
    # Spatial distribution of reflections
    ax4 = fig.add_subplot(gs[2, 0])
    delays = (distances / speeds * sample_rate).astype(int)
    ax4.hist(delays / sample_rate * 1000, bins=50, alpha=0.7)
    ax4.set_title('Distribution of Reflection Delays')
    ax4.set_xlabel('Delay (ms)')
    ax4.set_ylabel('Count')
    ax4.grid(True)
    
    # Channel energy distribution
    ax5 = fig.add_subplot(gs[2, 1])
    channel_energy = np.sum(ir**2, axis=1)
    ax5.bar(np.arange(len(channel_energy)), channel_energy)
    ax5.set_title('Channel Energy Distribution')
    ax5.set_xlabel('Channel Index')
    ax5.set_ylabel('Total Energy')
    ax5.set_yscale('log')
    ax5.grid(True)
    
    plt.tight_layout()
    return fig

def test_ambisonic_ir_from_gsound_data():
    df = pd.read_parquet("/root/pygsound-sir/output/20241216_004202_1x1_4000998paths.parquet")
    order = 7
    sample_rate = 48000
    
    listener_directions = np.vstack([
        df['listener_direction_x'], 
        df['listener_direction_y'], 
        df['listener_direction_z']
    ]).T.astype(np.float32)
    
    # Get intensities for all frequency bands
    intensity_columns = [col for col in df.columns if col.startswith('intensity_band_')]
    intensities = df[intensity_columns].to_numpy().astype(np.float32)
    
    distances = df['distance'].to_numpy().astype(np.float32)
    speeds = df['speed_of_sound'].to_numpy().astype(np.float32)
    path_types = np.ones(len(df), dtype=np.int32) # not used now
    
    frequency_points = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=np.float32)
    
    ir = sh.generate_ambisonic_ir(
        order=order,
        listener_directions=listener_directions,
        intensities=intensities,
        distances=distances,
        speeds=speeds,
        path_types=path_types,
        frequency_points=frequency_points,
        sample_rate=sample_rate,
        normalize=True
    )
    
    print("\nAmbisonic IR Properties:")
    print(f"Number of SH coefficients: {ir.shape[0]}")
    print(f"IR length in samples: {ir.shape[1]}")
    print(f"IR length in ms: {ir.shape[1]/sample_rate*1000:.2f}")

    channel_energy = np.sum(ir**2, axis=1)
    print("\nChannel Energy:")
    print(f"Total energy: {np.sum(channel_energy):.6f}")
    print(f"Energy distribution: {channel_energy}")

    
    print("\nIR Analysis:")
    print(f"Maximum amplitude: {np.max(np.abs(ir)):.6f}")
    print(f"Number of non-zero time points: {np.count_nonzero(np.any(ir != 0, axis=0))}")

    fig = analyze_and_visualize_ir(
        ir=ir,
        sample_rate=sample_rate,
        distances=distances,
        speeds=speeds,
        frequency_points=frequency_points
    )
    
    fig.savefig("ambisonic_ir_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("\nAmbisonic IR Analysis Complete:")
    print(f"- Figure saved as 'ambisonic_ir_analysis.png'")
    print(f"- IR shape: {ir.shape}")
    print(f"- Maximum amplitude: {np.max(np.abs(ir)):.6f}")
    print(f"- Number of unique delay points: {len(np.unique((distances / speeds * sample_rate).astype(int)))}")
    
    np.save("ambisonic_ir.npy", ir)
    print("- IR data saved to 'ambisonic_ir.npy'")

if __name__ == "__main__":
    test_ambisonic_ir_from_gsound_data()
