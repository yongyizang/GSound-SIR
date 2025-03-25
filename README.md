# GSound-SIR 
*Get Raw Room Spatial Impulse Response Ray Tracing Data*
Accepted at AES 2025 Europe

*(under active development)*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

[Paper Link](https://arxiv.org/abs/2503.17866)


**Important licensing information**: This repository is developed on the basis of GSound and PyGSound, which follow their respective license. See [license](https://github.com/GAMMA-UMD/pygsound/tree/master?tab=License-1-ov-file) for details.

We will gradually remove all depenedencies to PyGSound. All code added additionally are open-sourced under Apache 2.0 license.

## Thanks!!
Special thanks to Dr. Carl Schissler and Dr. Zhenyu Tang for very helpful email correspondance, and thanks to Xuzhou Ye and Junjie Shi for incredibly fruitful discussions that guided this project's direction.

## Getting Started
```bash
conda create -n gsoundsir python=3.10
conda activate gsoundsir
cd auralizer
pip install .
cd ../ray_generator
pip install .
```

## Overview
**GSound-SIR** is a Python-based extension of GSound designed for efficient room acoustics simulation and high-order Ambisonic impulse response (SIR) generation. While GSound provides a powerful C++ ray-tracing core for interactive applications, it traditionally acts as a black box—limiting users to final acoustic results. GSound-SIR changes that by exposing raw ray-level data, enabling in-depth inspection, custom analysis, and advanced post-processing.

**Key Highlights:**
- **Raw Ray Data Access**: Directly inspect propagation paths (arrival times, directions, energy, etc.) in Python.
- **Higher-Order Ambisonics**: Synthesize spatial impulse responses up to 9th-order Ambisonics.
- **Optimized Data I/O**: Efficiently store ray data in [Parquet](https://parquet.apache.org/) format for seamless integration with modern data analysis toolchains.
- **Energy-Based Filtering**: Keep only the most acoustically significant ray paths (e.g., top X\% by energy) to reduce storage overhead.
- **Pythonic Interface**: Built using [pybind11](https://github.com/pybind/pybind11) for easy integration into Python workflows.


## Background
Ray tracing is a widely-used technique in acoustic simulations and spatial audio research. It strikes a balance between:
- **Accuracy**: Capturing complex reflection/diffraction.
- **Efficiency**: Scaling to large or arbitrarily shaped environments more feasibly than many finite-difference methods.

Traditional open-source libraries often output final impulse responses (IRs) without providing access to intermediate ray-level data. GSound-SIR bridges this gap, allowing full inspection of each ray path, enabling advanced custom analysis and flexible post-processing.

## Features
- **Decoupled Ray Generation and Auralization**  
  Run the core C++ ray tracer to extract raw ray data, then process or auralize in Python.
- **Energy-Based Filtering**  
  Output only the top-X or top-X\% of highest-energy rays, reducing storage needs while preserving accuracy.
- **Parquet Output**  
  Export large datasets in columnar format for efficient reading and analysis in Python (pandas, PySpark, etc.).
- **Flexible Ambisonic Orders**  
  Convert raw rays into high-order Ambisonic impulse responses (up to 9th order). 
- **Extensible**  
  Implement your own custom weighting, filtering, or HRTF-based rendering, and efficiently call them in Python (See `auralizer/src/cpp/binding.cpp`)
   
## Performance Benchmarks
Empirical results demonstrate:
- **Linear Scaling with Ray Count**: Computation time grows linearly with diffuse + specular ray count.

- **Energy Concentration**: A small fraction of rays can carry the majority of acoustic energy (top 0.1% ~ 80+% energy in certain tests).
- **Disk I/O**: Writing large volumes of ray data is often the bottleneck. Parquet storage plus energy-based filtering can drastically reduce size and time.

## Limitations and Future Work
- **Stationary Sources**: Currently optimized for stationary sources. Moving source support is on our roadmap.
- **No GPU Acceleration**: CPU-bound but multi-threaded; GPU acceleration is a future enhancement.
- **Ambisonic Auralization**: Extensible to other rendering methods (e.g., binaural/HRTF-based) with custom Python scripts.
- **Potential for Deep Learning**: The concentrated nature of ray energy suggests that neural upsampling or ML-based reflection modeling is a promising direction.

## Citing GSound-SIR
If you use GSound-SIR in your work, we appreciate citations:

```
@misc{zang2025gsoundsirspatialimpulseresponse,
      title={GSound-SIR: A Spatial Impulse Response Ray-Tracing and High-order Ambisonic Auralization Python Toolkit}, 
      author={Yongyi Zang and Qiuqiang Kong},
      year={2025},
      eprint={2503.17866},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.17866}, 
}
```

You may also reference the underlying [GSound project]([https://github.com/jackpesch/gsound](https://github.com/GAMMA-UMD/pygsound/tree/master/src/GSound)) where appropriate.

## License
- **GSound-SIR Python Bindings and Extensions**: [Apache License 2.0](LICENSE)
- **Core GSound**: Distributed under its own license.

Please review both licenses before integrating GSound-SIR into your projects.

---

**Thank you for using GSound-SIR!** Feel free to open issues for bugs or feature requests, and we welcome pull requests. For deeper discussions or collaboration inquiries, please contact Yongyi Zang (zyy0116@gmail.com).
