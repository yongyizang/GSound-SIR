#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

/**
 * @brief Calculates the Spherical Harmonics normalization constant K(l, m) for N3D.
 *
 * K(l,m) = sqrt( (2*l + 1) * (l - |m|)! / (4 * PI * (l + |m|)!) )
 * The calculation is performed using logarithms to handle large factorials without overflow.
 *
 * @param l The SH order (degree).
 * @param m The SH index (degree).
 * @return The normalization constant as a double.
 */
double calculate_sh_normalization(unsigned int l, int m) {
    const unsigned int abs_m = std::abs(m);
    
    // Use log-gamma for factorial calculation to avoid overflow and precision issues.
    // lgamma(n) = log((n-1)!)
    double log_val = 0.5 * (
        std::log(2.0 * l + 1.0) - 
        std::log(4.0 * M_PI) + 
        std::lgamma(l - abs_m + 1.0) - 
        std::lgamma(l + abs_m + 1.0)
    );

    return std::exp(log_val);
}

/**
 * @brief Evaluates real-valued Spherical Harmonics (N3D/ACN ordering) for a given direction.
 *
 * This function calculates the SH coefficients up to a specified order for a Cartesian direction vector (x, y, z).
 * It uses the efficient recurrence-based method described by Sloan [2013], which avoids
 * explicit trigonometric function calls (sin, cos, atan2).
 *
 * @param max_order The maximum SH order to compute (e.g., 3 for third-order Ambisonics).
 * @param x The x-component of the normalized direction vector.
 * @param y The y-component of the normalized direction vector.
 * @param z The z-component of the normalized direction vector.
 * @param sh_norm_factors A pre-calculated table of normalization factors.
 * @param sh_coefficients A pre-allocated vector to store the output SH coefficients. Its size must be (max_order + 1)^2.
 */
void evaluate_spherical_harmonics(
    unsigned int max_order, 
    float x, float y, float z, 
    const std::vector<std::vector<double>>& sh_norm_factors,
    std::vector<float>& sh_coefficients
) {
    unsigned int num_coeffs = (max_order + 1) * (max_order + 1);
    if (sh_coefficients.size() != num_coeffs) {
        throw std::invalid_argument("Output vector size must match (max_order + 1)^2.");
    }

    // Pre-calculate Associated Legendre Polynomials with (sin(theta))^m factored out
    // This uses the recurrence relations from Sloan [2013], Eq. 4. https://jcgt.org/published/0002/02/06/
    std::vector<std::vector<double>> p_lm(max_order + 1, std::vector<double>(max_order + 1, 0.0));
    
    // P_0^0 = 1
    p_lm[0][0] = 1.0;
    // P_m^m = (1-2m) * P_{m-1}^{m-1}
    for (unsigned int m = 1; m <= max_order; ++m) {
        p_lm[m][m] = (1.0 - 2.0 * m) * p_lm[m-1][m-1];
    }

    // P_{m+1}^m(z) = z * (2m+1) * P_m^m(z)
    for (unsigned int m = 0; m < max_order; ++m) {
        p_lm[m+1][m] = z * (2.0 * m + 1.0) * p_lm[m][m];
    }
    
    // Calculate P_l^m(z) for l > m+1 using the main recurrence relation:
    // (l-m) * P_l^m = z * (2l-1) * P_{l-1}^m - (l+m-1) * P_{l-2}^m
    for (unsigned int l = 2; l <= max_order; ++l) {
        for (unsigned int m = 0; m < l - 1; ++m) {
            p_lm[l][m] = (z * (2.0 * l - 1.0) * p_lm[l-1][m] - (l + m - 1.0) * p_lm[l-2][m]) / (l - m);
        }
    }
    
    // --- Combine with normalization and azimuthal terms using recurrence on x,y ---
    double sqrt2 = std::sqrt(2.0);
    
    // m = 0 (zonal harmonics). These only depend on z.
    for (unsigned int l = 0; l <= max_order; ++l) {
        int acn = l * l + l;
        sh_coefficients[acn] = sh_norm_factors[l][0] * p_lm[l][0];
    }

    // m > 0 (tesseral/sectoral harmonics)
    // We use a recurrence to calculate the (sin(theta))^m * cos/sin(m*phi) terms,
    // which are pure polynomials of x and y.
    // Let A_m = (sin(theta))^m * cos(m*phi), B_m = (sin(theta))^m * sin(m*phi)
    // A_m = A_{m-1}*x - B_{m-1}*y
    // B_m = A_{m-1}*y + B_{m-1}*x
    double a_m_prev = 1.0; // A_0
    double b_m_prev = 0.0; // B_0

    for (unsigned int m = 1; m <= max_order; ++m) {
        // Update the azimuthal polynomial terms
        double a_m = a_m_prev * x - b_m_prev * y;
        double b_m = a_m_prev * y + b_m_prev * x;

        // Combine Legendre polynomials with azimuthal terms
        for (unsigned int l = m; l <= max_order; ++l) {
            int acn_pos = l * l + l + m; // Index for +m
            int acn_neg = l * l + l - m; // Index for -m
            
            double norm_factor = sh_norm_factors[l][m];
            double p_val = p_lm[l][m];

            sh_coefficients[acn_pos] = norm_factor * sqrt2 * p_val * a_m; // m > 0
            sh_coefficients[acn_neg] = norm_factor * sqrt2 * p_val * b_m; // m < 0
        }

        // Store current values for next iteration
        a_m_prev = a_m;
        b_m_prev = b_m;
    }
}

/**
 * @brief A simple pseudo-random number generator for creating white noise.
 */
class NoiseGenerator {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
    
public:
    NoiseGenerator(unsigned int seed = 42) : gen(seed), dist(-1.0f, 1.0f) {}
    
    float sample() {
        return dist(gen);
    }
};

/**
 * @brief A structure to hold multiple frequency bands for one audio sample.
 */
struct SIMDBands {
    std::vector<float> bands;
    SIMDBands(int numBands) : bands(numBands, 0.0f) {}
};

/**
 * @brief A Linkwitz-Riley crossover filter bank.
 */
class CrossoverFilter {
public:
    // State for a single biquad filter section.
    struct BiquadState {
        float z1 = 0.0f, z2 = 0.0f;
        void reset() { z1 = 0.0f; z2 = 0.0f; }
    };

    // Coefficients for a single biquad filter.
    struct BiquadCoeffs {
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f; // Feedforward
        float a1 = 0.0f, a2 = 0.0f;           // Feedback
    };

    CrossoverFilter(float sampleRate, const std::vector<float>& freqPoints) 
        : sampleRate(sampleRate)
        , numBands(freqPoints.size() + 1)
        , filterCoeffs(numBands)
        , filterStates(numBands)
    {
        // Design the parallel filter bank.
        for (size_t i = 0; i < numBands; ++i) {
            // Band 0 is the first low-pass filter.
            if (i == 0) {
                designLinkwitzRiley(freqPoints[i], filterCoeffs[i], false);
            } 
            // The last band is the final high-pass filter.
            else if (i == numBands - 1) {
                designLinkwitzRiley(freqPoints[i-1], filterCoeffs[i], true);
            }
            // Intermediate bands are band-pass filters.
            else {
                // A 4th-order bandpass is a cascade of a 4th-order HP and 4th-order LP
                filterCoeffs[i].resize(4);
                std::vector<BiquadCoeffs> hp_coeffs, lp_coeffs;
                designLinkwitzRiley(freqPoints[i-1], hp_coeffs, true);
                designLinkwitzRiley(freqPoints[i], lp_coeffs, false);
                filterCoeffs[i][0] = hp_coeffs[0];
                filterCoeffs[i][1] = hp_coeffs[1];
                filterCoeffs[i][2] = lp_coeffs[0];
                filterCoeffs[i][3] = lp_coeffs[1];
            }
            filterStates[i].resize(filterCoeffs[i].size());
        }
    }

    void reset() {
        for (auto& band_states : filterStates) {
            for (auto& state : band_states) {
                state.reset();
            }
        }
    }

    void process(const float* input, SIMDBands* output, int numSamples) {
        for (int i = 0; i < numSamples; ++i) {
            for (int band = 0; band < numBands; ++band) {
                float sample = input[i];
                for (size_t stage = 0; stage < filterCoeffs[band].size(); ++stage) {
                    sample = applyBiquad(sample, filterCoeffs[band][stage], filterStates[band][stage]);
                }
                output[i].bands[band] = sample;
            }
        }
    }

private:
    float sampleRate;
    int numBands;
    std::vector<std::vector<BiquadCoeffs>> filterCoeffs;
    std::vector<std::vector<BiquadState>> filterStates;

    void designLinkwitzRiley(float freq, std::vector<BiquadCoeffs>& coeffs, bool isHighpass) {
        coeffs.resize(2);
        float w0 = 2.0f * M_PI * freq / sampleRate;
        float cos_w0 = std::cos(w0);
        float sin_w0 = std::sin(w0);
        float alpha = sin_w0 / (2.0f * (1.0f / std::sqrt(2.0f))); // Q = 1/sqrt(2) for Butterworth

        float b0, b1, b2, a0, a1, a2;
        if (isHighpass) {
            b0 = (1.0f + cos_w0) / 2.0f;
            b1 = -(1.0f + cos_w0);
            b2 = (1.0f + cos_w0) / 2.0f;
        } else { // Lowpass
            b0 = (1.0f - cos_w0) / 2.0f;
            b1 = 1.0f - cos_w0;
            b2 = (1.0f - cos_w0) / 2.0f;
        }
        a0 = 1.0f + alpha;
        a1 = -2.0f * cos_w0;
        a2 = 1.0f - alpha;

        coeffs[0].b0 = b0 / a0;
        coeffs[0].b1 = b1 / a0;
        coeffs[0].b2 = b2 / a0;
        coeffs[0].a1 = a1 / a0;
        coeffs[0].a2 = a2 / a0;
        
        // 4th order is two identical 2nd order filters cascaded
        coeffs[1] = coeffs[0];
    }

    /**
     * @brief Applies a biquad filter using the stable Direct Form II Transposed structure.
     */
    float applyBiquad(float sample, const BiquadCoeffs& c, BiquadState& s) {
        float out = c.b0 * sample + s.z1;
        s.z1 = c.b1 * sample - c.a1 * out + s.z2;
        s.z2 = c.b2 * sample - c.a2 * out;
        return out;
    }
};

/**
 * @brief Generates a multi-channel Ambisonic impulse response based on acoustic path data.
 */
py::array_t<float> generate_ambisonic_ir(
    int order,
    py::array_t<float> listener_directions,
    py::array_t<float> intensities,
    py::array_t<float> distances,
    py::array_t<float> speeds,
    py::array_t<float> frequency_points,
    float sample_rate,
    bool precise_early_reflections = false,
    bool normalize = true,
    double early_reflection_threshold = 0.01
) {
    if (order < 0) {
        throw std::runtime_error("Order must be non-negative.");
    }
    
    auto directions_buf = listener_directions.unchecked<2>();
    auto intensities_buf = intensities.unchecked<2>();
    auto distances_buf = distances.unchecked<1>();
    auto speeds_buf = speeds.unchecked<1>();
    auto freq_buf = frequency_points.request();

    ssize_t num_paths = directions_buf.shape(0);
    ssize_t num_bands = intensities_buf.shape(1);
    const unsigned int u_order = order;
    int num_coefficients = (u_order + 1) * (u_order + 1);
    
    std::vector<float> freq_points(static_cast<float*>(freq_buf.ptr), static_cast<float*>(freq_buf.ptr) + freq_buf.size);
    if (freq_points.size() != static_cast<size_t>(num_bands - 1)) {
        throw std::runtime_error("Number of frequency points must be number of bands - 1.");
    }
    
    // Pre-calculate SH Normalization Factors
    // These factors are constant for a given order, so we compute them once
    // to avoid redundant calculations inside the main processing loops.
    std::vector<std::vector<double>> sh_norm_factors(u_order + 1, std::vector<double>(u_order + 1));
    for (unsigned int l = 0; l <= u_order; ++l) {
        for (unsigned int m = 0; m <= l; ++m) {
            sh_norm_factors[l][m] = calculate_sh_normalization(l, m);
        }
    }

    // Calculate IR Length
    float max_delay = 0.0f;
    for (ssize_t i = 0; i < num_paths; ++i) {
        max_delay = std::max(max_delay, distances_buf(i) / speeds_buf(i));
    }
    // Define filter padding as a named constant for clarity.
    const int FILTER_TAIL_PADDING = 2048;
    int ir_length = static_cast<int>(std::ceil(max_delay * sample_rate)) + FILTER_TAIL_PADDING;

    // Create Noise and Output Buffers
    std::vector<float> raw_noise(ir_length);
    NoiseGenerator noise_gen;
    for (int i = 0; i < ir_length; ++i) {
        raw_noise[i] = noise_gen.sample();
    }
    
    CrossoverFilter crossover(sample_rate, freq_points);
    std::vector<SIMDBands> filtered_noise(ir_length, SIMDBands(num_bands));
    crossover.process(raw_noise.data(), filtered_noise.data(), ir_length);
    
    std::vector<ssize_t> output_shape = {static_cast<ssize_t>(num_coefficients), static_cast<ssize_t>(ir_length)};
    auto result = py::array_t<float>(output_shape);
    auto result_buf = result.mutable_unchecked<2>();
    std::fill(result.mutable_data(), result.mutable_data() + result.size(), 0.0f);

    // Path Separation
    std::vector<int> late_reflection_indices;
    std::vector<int> early_reflection_indices;
    std::vector<double> path_energies(num_paths, 0.0);

    if (num_paths > 0) {
        double total_energy = 0;
        for(int path = 0; path < num_paths; ++path) {
            for(int band = 0; band < num_bands; ++band) {
                path_energies[path] += intensities_buf(path, band);
            }
            total_energy += path_energies[path];
        }

        if (precise_early_reflections) { // Separate paths into early and late reflections based on energy threshold
            const double energy_threshold_value = total_energy * early_reflection_threshold;
            for(int path = 0; path < num_paths; ++path) {
                if (path_energies[path] > energy_threshold_value && total_energy > 0) {
                    early_reflection_indices.push_back(path);
                } else {
                    late_reflection_indices.push_back(path);
                }
            }
        } else {
            late_reflection_indices.resize(num_paths);
            std::iota(late_reflection_indices.begin(), late_reflection_indices.end(), 0);
        }
    }
    
    // Synthesize Late Reflections
    if (!late_reflection_indices.empty()) {
        // Create and populate histograms for the late field
        std::vector<std::vector<float>> binned_energy_per_band(num_bands, std::vector<float>(ir_length, 0.0f));
        std::vector<std::vector<float>> binned_sh_sum(num_coefficients, std::vector<float>(ir_length, 0.0f));
        std::vector<float> binned_total_energy(ir_length, 0.0f);
        std::vector<float> sh_coeffs(num_coefficients);

        for (int path_idx : late_reflection_indices) {
            float delay = distances_buf(path_idx) / speeds_buf(path_idx);
            int delay_samples = static_cast<int>(std::floor(delay * sample_rate));
            if (delay_samples >= ir_length) continue;

            // Accumulate energy per band for this time bin
            for (ssize_t band = 0; band < num_bands; ++band) {
                binned_energy_per_band[band][delay_samples] += intensities_buf(path_idx, band);
            }
            
            // Accumulate total energy and energy-weighted SH for this time bin
            float total_path_energy = path_energies[path_idx];
            binned_total_energy[delay_samples] += total_path_energy;
            
            float dx = directions_buf(path_idx, 0), dy = directions_buf(path_idx, 1), dz = directions_buf(path_idx, 2);
            float length = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (length > 1e-6f) { dx /= length; dy /= length; dz /= length; }
            evaluate_spherical_harmonics(u_order, dx, dy, dz, sh_norm_factors, sh_coeffs);
            
            for (int coeff = 0; coeff < num_coefficients; ++coeff) {
                binned_sh_sum[coeff][delay_samples] += sh_coeffs[coeff] * total_path_energy;
            }
        }

        // Synthesize audio from the populated histograms
        std::vector<float> normalized_sh_for_bin(num_coefficients);
        for (int t = 0; t < ir_length; ++t) {
            if (binned_total_energy[t] < 1e-9f) continue;

            // Calculate the energy-weighted average SH for this time bin
            for (int coeff = 0; coeff < num_coefficients; ++coeff) {
                normalized_sh_for_bin[coeff] = binned_sh_sum[coeff][t] / binned_total_energy[t];
            }

            // Calculate the pressure-weighted noise sample from all bands
            float weighted_noise_sample = 0.0f;
            for (ssize_t band = 0; band < num_bands; ++band) {
                float pressure_envelope = std::sqrt(binned_energy_per_band[band][t]);
                weighted_noise_sample += pressure_envelope * filtered_noise[t].bands[band];
            }

            // Apply the spatial information to the noisy sample and add to result
            for (int coeff = 0; coeff < num_coefficients; ++coeff) {
                result_buf(coeff, t) += normalized_sh_for_bin[coeff] * weighted_noise_sample;
            }
        }
    }

    // Synthesize Early Reflections (Discrete)
    if (precise_early_reflections && !early_reflection_indices.empty()) {
        CrossoverFilter impulse_crossover(sample_rate, freq_points);
        std::vector<float> sh_coeffs(num_coefficients);
        
        for (int path_idx : early_reflection_indices) {
            float delay = distances_buf(path_idx) / speeds_buf(path_idx);
            int delay_samples = static_cast<int>(std::floor(delay * sample_rate));
            if (delay_samples >= ir_length) continue;

            float dx = directions_buf(path_idx, 0), dy = directions_buf(path_idx, 1), dz = directions_buf(path_idx, 2);
            float length = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (length > 1e-6f) { dx /= length; dy /= length; dz /= length; }
            evaluate_spherical_harmonics(u_order, dx, dy, dz, sh_norm_factors, sh_coeffs);

            int impulse_len = ir_length - delay_samples;
            std::vector<SIMDBands> impulse_response(impulse_len, SIMDBands(num_bands));
            std::vector<float> raw_impulse(impulse_len, 0.0f);
            raw_impulse[0] = 1.0f; // dirac

            impulse_crossover.reset();
            impulse_crossover.process(raw_impulse.data(), impulse_response.data(), impulse_len);

            for (int i = 0; i < impulse_len; ++i) {
                float weighted_impulse_sample = 0.0f;
                 for (ssize_t band = 0; band < num_bands; ++band) {
                    weighted_impulse_sample += std::sqrt(intensities_buf(path_idx, band)) * impulse_response[i].bands[band];
                }
                for (int coeff = 0; coeff < num_coefficients; ++coeff) {
                    // Add to the existing late-field result
                    result_buf(coeff, i + delay_samples) += sh_coeffs[coeff] * weighted_impulse_sample;
                }
            }
        }
    }
    
    // Normalization
    if (normalize) {
        float max_sample = 0.0f;
        for (int coeff = 0; coeff < num_coefficients; ++coeff) {
            for (int i = 0; i < ir_length; ++i) {
                max_sample = std::max(max_sample, std::abs(result_buf(coeff, i)));
            }
        }
        
        if (max_sample > 1e-6f) {
            float scale = 1.0f / max_sample;
            for (int coeff = 0; coeff < num_coefficients; ++coeff) {
                for (int i = 0; i < ir_length; ++i) {
                    result_buf(coeff, i) *= scale;
                }
            }
        }
    }
    
    return result;
}

PYBIND11_MODULE(spherical_harmonics_rt, m) {
    m.doc() = "Runtime Spherical Harmonics processor for generating Ambisonic IR waveforms (Optimized)";
    
    m.def("generate_ambisonic_ir", &generate_ambisonic_ir,
          "Generate Ambisonic IR waveforms using noise-based synthesis",
          py::arg("order"),
          py::arg("listener_directions"),
          py::arg("intensities"), 
          py::arg("distances"),
          py::arg("speeds"),
          py::arg("frequency_points"),
          py::arg("sample_rate"),
          py::arg("precise_early_reflections") = false,
          py::arg("normalize") = true,
          py::arg("early_reflection_threshold") = 0.01,
          py::return_value_policy::take_ownership);
}
