# Spherical Harmonics Auralizer
An efficient way of calculting spherical harmonics, in Python.

# Installing
```bash
pip install .
```

# Minimal usage example
```python
import numpy as np
import spherical_harmonics as sh

order = 3  # Ambisonic order
sample_rate = 48000  # Hz

# Simulated listener directions
num_directions = 100
listener_directions = np.random.randn(num_directions, 3).astype(np.float32)
listener_directions /= np.linalg.norm(listener_directions, axis=1, keepdims=True)

# Simulated intensities for a single frequency band
intensities = np.random.rand(num_directions, 1).astype(np.float32)

# Simulated distances and speeds
distances = np.random.uniform(1.0, 10.0, size=num_directions).astype(np.float32)
speeds = np.full(num_directions, 343.0, dtype=np.float32)

# Simulated path types (not used right now during IR synthesis)
path_types = np.ones(num_directions, dtype=np.int32)

# Define frequency points (Hz)
frequency_points = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=np.float32)

# Generate ambisonic IR
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
```

# Using with PyGSound-SIR
```python
df = pd.read_parquet("/root/pygsound-sir/output/20241216_004202_1x1_4000998paths.parquet")
    
order = 7
sample_rate = 48000

listener_directions = np.vstack([
    df['listener_x'], 
    df['listener_y'], 
    df['listener_z']
]).T.astype(np.float32)

intensity_columns = [col for col in df.columns if col.startswith('intensity_band_')]
intensities = df[intensity_columns].to_numpy().astype(np.float32)

distances = df['distance'].to_numpy().astype(np.float32)
speeds = df['speed_of_sound'].to_numpy().astype(np.float32)
path_types = np.ones(len(df), dtype=np.int32)

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
```
