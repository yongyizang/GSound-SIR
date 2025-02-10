import numpy as np
import pygsound as ps
import pandas as pd
from datetime import datetime
import os
from typing import Dict, List, Tuple, Any, Optional
from multiprocessing import Pool, cpu_count
from itertools import product

def process_position_pair(args):
    """
    Worker function to process a single source-listener position pair.
    Returns data in a format ready for DataFrame conversion.
    """
    mesh_path, src_pos, lis_pos, params, timestamp = args

    ctx = ps.Context()
    ctx.diffuse_count = params['diffuse_count']
    ctx.specular_count = params['specular_count']
    ctx.channel_type = ps.ChannelLayoutType.stereo

    mesh = ps.loadobj(mesh_path)
    scene = ps.Scene()
    scene.setMesh(mesh)

    src = ps.Source(src_pos)
    src.radius = params['source_radius']
    src.power = params['source_power']
    
    lis = ps.Listener(lis_pos)
    lis.radius = params['listener_radius']
    
    path_data = scene.getPathData(
        [src], [lis], ctx,
        energy_percentage=params['energy_percentage'],
        max_rays=params['max_rays']
    )["path_data"][0]
    
    print(f"Processed source-listener pair: {src_pos} -> {lis_pos}")
    params['timestamp'] = timestamp
    params['num_paths'] = path_data['num_paths']
    params['num_bands'] = path_data['num_bands']
    params['total_energy'] = path_data['total_energy']
    params['kept_energy_percentage'] = path_data['kept_energy_percentage']

    data_dict = {
        'source_x': [src_pos[0]] * path_data['num_paths'],
        'source_y': [src_pos[1]] * path_data['num_paths'],
        'source_z': [src_pos[2]] * path_data['num_paths'],
        'listener_x': [lis_pos[0]] * path_data['num_paths'],
        'listener_y': [lis_pos[1]] * path_data['num_paths'],
        'listener_z': [lis_pos[2]] * path_data['num_paths'],
        'source_direction_x': path_data['source_directions'][:, 0],
        'source_direction_y': path_data['source_directions'][:, 1],
        'source_direction_z': path_data['source_directions'][:, 2],
        'distance': path_data['distances'],
        'relative_speed': path_data['relative_speeds'],
        'speed_of_sound': path_data['speeds_of_sound'],
    }

    intensities_df = pd.DataFrame(
        path_data['intensities'],
        columns=[f'intensity_band_{i}' for i in range(path_data['num_bands'])]
    )

    df = pd.DataFrame(data_dict)
    df = pd.concat([df, intensities_df], axis=1)
    for param_name, param_value in params.items():
        df[f'param_{param_name}'] = param_value
        
    return df

class RayDataPipeline:
    def __init__(self, 
                 diffuse_count: int = 20000,
                 specular_count: int = 2000,
                 source_radius: float = 0.01,
                 source_power: float = 1.0,
                 listener_radius: float = 0.01,
                 energy_percentage: float = 100.0,
                 max_rays: int = 0,
                 num_workers: Optional[int] = None):
        """
        Initialize the ray tracing pipeline with configurable parameters.
        
        Args:
            diffuse_count: Number of diffuse rays
            specular_count: Number of specular rays
            source_radius: Radius of sound sources
            source_power: Power of sound sources
            listener_radius: Radius of listeners
            energy_percentage: Percentage of total energy to keep (0-100)
            max_rays: Maximum number of rays to keep (0 for no limit)
            num_workers: Number of parallel workers (None for CPU count)
        """
        self.params = {
            'diffuse_count': diffuse_count,
            'specular_count': specular_count,
            'source_radius': source_radius,
            'source_power': source_power,
            'listener_radius': listener_radius,
            'energy_percentage': energy_percentage,
            'max_rays': max_rays
        }
        
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        self.ctx = ps.Context()
        self.ctx.diffuse_count = diffuse_count
        self.ctx.specular_count = specular_count
        self.ctx.channel_type = ps.ChannelLayoutType.stereo

    def process_coordinates(self, 
                          mesh_path: str,
                          source_positions: List[Tuple[float, float, float]],
                          listener_positions: List[Tuple[float, float, float]],
                          output_path: str) -> str:
        """
        Process all source-listener position pairs and save results.
        
        Args:
            mesh_path: Path to the mesh file
            source_positions: List of source position tuples (x, y, z)
            listener_positions: List of listener position tuples (x, y, z)
            output_path: Directory to save the output file
            
        Returns:
            Path to the output file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path:
            os.makedirs(output_path, exist_ok=True)

        work_items = [
            (mesh_path, src_pos, lis_pos, self.params, timestamp)
            for src_pos, lis_pos in product(source_positions, listener_positions)
        ]

        # Process in parallel and collect DataFrames
        with Pool(processes=self.num_workers) as pool:
            dfs = pool.map(process_position_pair, work_items)
        
        # Combine all DataFrames
        final_df = pd.concat(dfs, ignore_index=True)

        # Include energy filtering info in filename
        energy_info = f"_e{self.params['energy_percentage']:.0f}" if self.params['energy_percentage'] < 100 else ""
        rays_info = f"_r{self.params['max_rays']}" if self.params['max_rays'] > 0 else ""
        
        output_filename = os.path.join(
            output_path,
            f"{timestamp}_{len(source_positions)}x{len(listener_positions)}{energy_info}{rays_info}_{len(final_df)}paths.parquet"
        )
        
        # Save to parquet format for better performance and compression
        final_df.to_parquet(output_filename, index=False)
                    
        print(f"Processing complete. Output saved to: {output_filename}")
        print(f"Processed {len(work_items)} source-listener pairs using {self.num_workers} workers")
        print(f"Total paths processed: {len(final_df)}")
        print(f"Average paths per pair: {len(final_df) / len(work_items):.1f}")
        print(f"Average kept energy: {final_df['param_kept_energy_percentage'].mean():.1f}%")
        
        return output_filename

def main():
    pipeline = RayDataPipeline(
        diffuse_count=20000,
        specular_count=2000,
        source_radius=0.01,
        source_power=1.0,
        listener_radius=0.01,
        energy_percentage=95.0,  # Keep 95% of total energy
        # max_rays=1000,          # But no more than 1000 rays
        num_workers=4
    )
    
    source_positions = [
        (1.0, 1.0, 0.5),
        (2.0, 2.0, 0.5)
    ]
    
    listener_positions = [
        (5.0, 3.0, 0.5),
        (4.0, 4.0, 0.5)
    ]
    
    pipeline.process_coordinates(
        mesh_path="/root/pygsound-sir/examples/cube.obj",
        source_positions=source_positions,
        listener_positions=listener_positions,
        output_path="output"
    )

if __name__ == '__main__':
    main()