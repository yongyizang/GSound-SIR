from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftshift
import os
from scipy.signal import convolve
from scipy.special import sph_harm
import torch
import trimesh

np.random.seed(0)
torch.manual_seed(0)

"""
reference:
    [1]:'Physically Based Real-Time Auralization of Interactive Virtual Environments'
    [2]:'https://github.com/qiuqiangkong/ism/blob/main/hoa.py
"""


def obj_volume_trimesh(
    obj_path: str,
    *,
    require_watertight: bool = False,
    fix_normals: bool = True,
    process: bool = True,
) -> float:
    """
    Load an OBJ with trimesh and return its volume.

    Args:
        obj_path: Path to .obj file.
        require_watertight: If True, raise if the mesh isn't watertight (recommended).
        fix_normals: If True, attempt to fix winding/normals for consistent orientation.
        process: Pass-through to trimesh.load (merges vertices, removes degenerate faces, etc).

    Returns:
        Volume (float) in units^3.

    Notes:
        - If your OBJ contains a Scene (multiple meshes), volumes are summed.
    """
    loaded = trimesh.load(obj_path, force=None, process=process)

    def _mesh_volume(m: "trimesh.Trimesh") -> float:
        if fix_normals:
            # Make winding/normals consistent (helps signed-volume correctness)
            trimesh.repair.fix_inversion(m)
            trimesh.repair.fix_normals(m)

        if require_watertight and not m.is_watertight:
            raise ValueError(
                f"Mesh is not watertight; volume may be invalid. "
                f"watertight={m.is_watertight}, euler_number={m.euler_number}"
            )

        # mesh.volume is already absolute volume for Trimesh
        return float(m.volume)

    # trimesh.load may return Trimesh or Scene depending on file content
    if isinstance(loaded, trimesh.Trimesh):
        return _mesh_volume(loaded)

    if isinstance(loaded, trimesh.Scene):
        total = 0.0
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                total += _mesh_volume(geom)
        if total == 0.0 and require_watertight:
            # could be empty or all non-mesh geometry
            raise ValueError("No mesh geometry found in OBJ/Scene.")
        return float(total)

    raise TypeError(f"Unsupported trimesh load result type: {type(loaded)}")

def _cart2sph(xyz,eps: float = 1e-12):
    """
        xyz: cartesian coordinates (3,*)
        return:
            theta: polar angle [0, pi]
            phi: azimuthal angle [0, 2*pi]
            in Radian system
    """
    xyz = np.asarray(xyz)
    if xyz.shape[0] == 3 and xyz.ndim == 2:
        x, y, z = xyz[0], xyz[1], xyz[2]
    elif xyz.shape[-1] == 3 and xyz.ndim == 2:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    else:
        raise ValueError(f"Expected shape (3,N) or (N,3), got {xyz.shape}")

    r = np.sqrt(x*x + y*y + z*z)
    # avoid division by zero for r=0
    z_over_r = np.clip(z / (r + eps), -1.0, 1.0)

    theta = np.arccos(z_over_r)      # [0, pi]
    phi = np.arctan2(y, x) + np.pi           # [0, 2*pi]
    doa = np.stack([theta, phi], axis=0)
    return doa

def create_dataset(ray, room):
    """
        return:
            a list of data_dict which can be used to initialize a dataset
    """
    ray_df = pd.read_parquet(ray)
    #sort by distance
    ray_df = ray_df.sort_values(by='distance', ascending=True)

    volumn = obj_volume_trimesh(room)
    c = np.mean(ray_df['speed_of_sound'].unique().tolist())
    num_bands = ray_df['param_num_bands'].unique()
    assert len(num_bands) == 1, "only one num_bands is supported"
    num_bands = num_bands[0]
    cols = ['source_x', 'source_y', 'source_z', 'listener_x', 'listener_y', 'listener_z']
    
    rx_tx_pairs_df = ray_df.groupby(cols)
    data_dict = []
    I_bands_cols = [f'intensity_band_{i}' for i in range(num_bands)]
    doa_cols = ['source_direction_x', 'source_direction_y', 'source_direction_z']
    for rx_tx_pair, pair_df in rx_tx_pairs_df:
        tx = np.array(rx_tx_pair[:3])
        rx = np.array(rx_tx_pair[3:6])
        Intensities = pair_df[I_bands_cols].to_numpy().transpose(-1,-2)#(num_bands, num_rays)
        Intensities = Intensities / num_bands
        doa_xyz = pair_df[doa_cols].to_numpy().transpose(-1,-2)#(3, num_rays)
        doa = _cart2sph(doa_xyz) 
        dist = pair_df['distance'].to_numpy()
        delay = dist / c #second
        data_dict.append({
            'tx': tx,
            'rx': rx,
            'Intensities': Intensities,
            'doa': doa,
            'delay': delay,
            'V': volumn
        })
    return data_dict

class Ambisonic_IR_Generator:
    def __init__(self, 
        fs=16000,
        order=1,
        imp_res_time = 1.,
        nfft = 8192,
        F_Vect = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=np.float32),
        c = 343,
        hist_time_step = 0.0040,
        ):
        self.fs = fs
        self.order = order
        self.imp_res_time = imp_res_time
        self.nfft = nfft
        self.c = c
        self.F_Vect = F_Vect
        self.hist_time_step = hist_time_step
        self.t_bins = round(imp_res_time / hist_time_step)
        self.f_bins = len(F_Vect)
        bef = self._get_bandedge_frequencies()
        self.BW = np.diff(bef)
    def _poisson_process(self, V):
        """
            generate poisson process according to the room size
            refer to [1] 5.3.4
        """
        t0 = ((2 * V * np.log(2)) / (4 * np.pi * self.c**3))**(1/3)
        poisson_process = []
        time_points = []
        t = t0
        while t < self.imp_res_time:
            time_points.append(t)
            
            # determine polarity
            if (np.round(t * self.fs) - t * self.fs) < 0:
                poisson_process.append(1)
            else:
                poisson_process.append(-1)
            
            # determine average event rate
            mu = min(1e4, 4 * np.pi * self.c**3 * t**2 / V)
            
            # determine interval size
            delta_ta = (1 / mu) * np.log(1 / np.random.rand())
            t += delta_ta
        # create sampled random process
        rand_seq = np.zeros(int(np.ceil(self.imp_res_time * self.fs)))
        for i, time_val in enumerate(time_points):
            idx = int(np.round(time_val * self.fs))
            if idx < len(rand_seq):
                rand_seq[idx] = poisson_process[i]
        return rand_seq

    def _bandpass_filter(self):
        """
            create raised cosine filter according to freq bands
            as a bandpass filter for noise
        """
        flow = self.F_Vect / 2
        fhigh = self.F_Vect * 2
        
        # frequency vector
        F = np.fft.fftfreq(self.nfft, 1/self.fs)
        F = F[:self.nfft//2 + 1]  # single-sided spectrum
        
        # create bandpass filters
        RCF = np.zeros((self.f_bins, len(F)))
        
        for i in range(self.f_bins):
            for j, f in enumerate(F):
                if f < self.F_Vect[i] and f >= flow[i]:
                    RCF[i, j] = 0.5 * (1 + np.cos(2 * np.pi * f / self.F_Vect[i]))
                elif f < fhigh[i] and f >= self.F_Vect[i]:
                    RCF[i, j] = 0.5 * (1 - np.cos(2 * np.pi * f / (2 * self.F_Vect[i])))
                else:
                    RCF[i, j] = 0
        
        # convert to time domain
        RCF_full = np.zeros((self.f_bins, self.nfft))
        RCF_full[:, :self.nfft//2 + 1] = RCF
        RCF_full[:, self.nfft//2 + 1:] = RCF[:, -2:0:-1]  # mirror symmetry
        RCF_time = np.real(fftshift(ifft(RCF_full, axis=1), axes=1))
        return RCF_time

    def _bandwidth_noise(self, V):
        """
            get bandwidth noise according to the room size
            hold true for all tx_rx_pairs in the same room
        """
        noise = self._poisson_process(V)
        RCF = self._bandpass_filter()
        bw_noise = np.zeros((self.f_bins, len(noise)))
        for i in range(self.f_bins):
            bw_noise[i,:] = convolve(noise, RCF[i,:], mode='same')
        return bw_noise#(num_bands, imp_res_time*fs)
    
    def _get_bandedge_frequencies(self):
        """
            get bandedge frequencies
            refer to [1] 5.3.4
        """
        G = 2
        bands_per_octave = 1
        fbpo = 0.5 / bands_per_octave
        
        bef = np.zeros(self.f_bins + 1)
        bef[0] = self.F_Vect[0] * G**(-fbpo)
        
        for i in range(self.f_bins - 1):
            bef[i + 1] = np.sqrt(self.F_Vect[i] * self.F_Vect[i + 1])
            bef[-1] = min(self.F_Vect[-1] * G**fbpo, self.fs/2)  # assume
        
        return bef

    def _get_Spherical_Harmonics(self, order, theta, phi):
        r"""
            refer to [2]
            Calculate the HOA coefficients a_nm of signals.
            a_nm = \int_{f(θ, φ) Y_nm^*(θ, φ) sinθ dθdφ}

            Args:
                order: HOA order
                theta: polar angle [0, pi]
                phi: azimuthal angle [0, 2pi]
            Outputs:
                a_nm: (c,1,1) where c = (order+1)^2
        """
        
        # Y_nm(θ, φ)
        bases = []
        for n in range(order + 1):
            for m in range(-n, n + 1):
                Y = sph_harm(m, n, phi, theta) 
                bases.append(Y.real) 
        
        # Y_nm(θ, φ)
        bases = np.stack(bases, axis=0)  # (c,)

        # f(θ, φ) Y_nm(θ, φ)
        hoa = bases  # (c)
        # Integral is dismissed because f(θ, φ) is a dirac delta function for ray
        return hoa[:,None]

    def _get_Hist(self, Intensities, delay, doa):
        """
            args:
                Intensities: (num_bands, num_rays)
                delay: (num_rays)
                doa: (2, num_rays)
            return:
                SH_Hist: (c, num_bands, t_bins) use the hoa of the maximum intensity to update the SH_Hist
                Egy_Hist: (num_bands, t_bins)
        """
        num_channels = (self.order + 1)**2
        num_rays = Intensities.shape[1]
        #SH_Hist: (c, num_bands, t_bins)
        SH_Hist = np.ones((num_channels, self.f_bins, self.t_bins))
        #Egy_Hist: (num_bands, t_bins)
        Egy_Hist = np.zeros((self.f_bins, self.t_bins))
        #Int_Hist: (num_rays)
        Int_Hist = np.zeros(num_rays)

        for i in range(num_rays):
            t_bin = int(round(delay[i]/self.hist_time_step))
            theta,phi = doa[:,i]
            if t_bin < self.t_bins:
                hoa = self._get_Spherical_Harmonics(self.order, theta, phi)
                Egy_Hist[:,t_bin] += Intensities[:,i]
                #use the hoa of the maximum intensity to update the SH_Hist
                if np.sum(Intensities[:,i]) > Int_Hist[i]:
                    Int_Hist[i] = np.sum(Intensities[:,i])
                    SH_Hist[:,:,t_bin] = hoa
        return SH_Hist, Egy_Hist
                
    def _amp_weighted_sum(self, y, Egy_Hist, SH_Hist):
        """
            refer to [1] 5.3.4
            args:
                y: bandwidth noise (num_bands, len(imptimes))
                Egy_Hist: Energy histogram (num_bands, t_bins)
                SH_Hist: Spherical harmonics histogram (c, num_bands, t_bins)
        """
        BW = self.BW
        imptimes = np.arange(y.shape[-1])/self.fs
        W = np.zeros((self.f_bins, len(imptimes)))
        SH_W = np.zeros((SH_Hist.shape[0], self.f_bins, len(imptimes)))
        for k in range(self.t_bins):
            gklow = int(np.floor(k*self.fs*self.hist_time_step))
            gkhigh = int(np.floor((k+1)*self.fs*self.hist_time_step))
            yy = y[:, gklow:gkhigh]**2
            e_value = np.sqrt(Egy_Hist[:,k]/np.sum(yy, axis = 1)) * np.sqrt(BW/(self.fs/2))#(num_bands, 1)
            W[:,gklow:gkhigh] = e_value[:,None]
            SH_W[:,:,gklow:gkhigh] = SH_Hist[:,:,k:k+1]
        y = y * W #(num_bands, len(imptimes))
        y = y[None,:] * SH_W #(c, num_bands, len(imptimes))
        y = np.sum(y, axis = 1) #(c, len(imptimes))
        return y

    def forward_ambsonics(self, data):
        tx = data['tx']
        rx = data['rx']
        Intensities = data['Intensities']
        delay = data['delay']
        doa = data['doa']
        V = data['V']
        # generate bandwidth noise
        y = self._bandwidth_noise(V)#(num_bands, imp_res_time*fs)
        # compute Hist
        SH_Hist, Egy_Hist = self._get_Hist(Intensities, delay, doa)
        #weighted sum
        y = self._amp_weighted_sum(y, Egy_Hist, SH_Hist)
        return y
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    def plot_sir(sir):
        num_samples = sir.shape[-1]
        num_channels = sir.shape[0]

        sampling_rate = 16000

        plt.figure(figsize=(10, 6))

        colors = cm.viridis(np.linspace(0, 1, num_channels))  # 使用viridis调色板

        for i in range(num_channels):
            plt.plot(np.arange(num_samples) / sampling_rate, sir[i,:], label=f'Channel {i+1}', color=colors[i])

        plt.title(f'{num_channels}-channel Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig('sir.png')

    ray = ''
    room = ''
    data = create_dataset(ray, room)
    demo = data[0]
    tx = demo['tx']
    rx = demo['rx']
    Intensities = demo['Intensities']
    doa = demo['doa']
    delay = demo['delay']
    V = demo['V']
    print(f"tx:{tx.shape}")
    print(f"rx:{rx.shape}")
    print(f"Intensities:{Intensities.shape}")
    print(f"doa:{doa.shape}")
    print(f"delay:{delay.shape}")
    print(f"V:{V}")
    auralizer = Ambisonic_IR_Generator()
    import time
    start_time = time.time()
    y = auralizer.forward_ambsonics(demo)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"y:{y.shape}")
    plot_sir(y)




    

        