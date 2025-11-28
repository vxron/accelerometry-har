#include "FeatureExtractor.hpp"
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// ====================== BASIC STAT HELPERS (forward decls) ====================

static float mean_vec(const std::vector<float>& v);
static float std_vec(const std::vector<float>& v);
static float rms_vec(const std::vector<float>& v);
static float mad_vec(const std::vector<float>& v);
static float iqr_vec(std::vector<float> v);
static float corr_vec(const std::vector<float>& a,
                      const std::vector<float>& b);

// ==== mag / PSD / band / entropy / AR forward decls ====
struct Ar4Result {
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    float c4 = 0.0f;
    float resvar = 0.0f;
};

static const std::vector<float>& mag_from_xyz(FeatureCache_S& ctx);
static void compute_psd_if_needed(FeatureCache_S& ctx);
static float bandpower_mag(FeatureCache_S& ctx, float lo, float hi);
static float total_power_mag(FeatureCache_S& ctx);
static float spectral_entropy_mag(FeatureCache_S& ctx);
static float dom_freq_mag(FeatureCache_S& ctx);
static float spec_centroid_mag(FeatureCache_S& ctx);
static float spec_bandwidth_mag(FeatureCache_S& ctx);
static float spec_flatness_mag(FeatureCache_S& ctx);
static Ar4Result ar4_mag(FeatureCache_S& ctx);
static float sma_xyz(FeatureCache_S& ctx);
static float mean_mag(FeatureCache_S& ctx);
static float std_mag(FeatureCache_S& ctx);
static float rms_mag(FeatureCache_S& ctx);
static float mad_mag(FeatureCache_S& ctx);
static float iqr_mag(FeatureCache_S& ctx);
static float ptp_mag(FeatureCache_S& ctx);

// STATIC FEATURE REGISTRY TO MAP JSON feat_names in meta to function calls (FeatureKind) for extraction.
namespace {
    const std::unordered_map<std::string, FeatureKind_E> g_feature_registry = {
        // Per-axis stats
        {"mean_x", FeatureKind_E::MeanAxisX},
        {"mean_y", FeatureKind_E::MeanAxisY},
        {"mean_z", FeatureKind_E::MeanAxisZ},

        {"std_x",  FeatureKind_E::StdAxisX},
        {"std_y",  FeatureKind_E::StdAxisY},
        {"std_z",  FeatureKind_E::StdAxisZ},

        {"rms_x",  FeatureKind_E::RmsAxisX},
        {"rms_y",  FeatureKind_E::RmsAxisY},
        {"rms_z",  FeatureKind_E::RmsAxisZ},

        {"mad_x",  FeatureKind_E::MadAxisX},
        {"mad_y",  FeatureKind_E::MadAxisY},
        {"mad_z",  FeatureKind_E::MadAxisZ},

        {"iqr_x",  FeatureKind_E::IqrAxisX},
        {"iqr_y",  FeatureKind_E::IqrAxisY},
        {"iqr_z",  FeatureKind_E::IqrAxisZ},

        // Magnitude/time-domain
        {"sma_xyz", FeatureKind_E::SmaXYZ},

        {"mean_mag", FeatureKind_E::MeanMag},
        {"std_mag",  FeatureKind_E::StdMag},
        {"rms_mag",  FeatureKind_E::RmsMag},
        {"mad_mag",  FeatureKind_E::MadMag},
        {"iqr_mag",  FeatureKind_E::IqrMag},

        // PSD / bandpower / entropy
        {"total_power_mag",      FeatureKind_E::TotalPowerMag},
        {"bp_0p5_3",             FeatureKind_E::Bp0p5_3},
        {"bp_3_8",               FeatureKind_E::Bp3_8},
        {"bp_8_15",              FeatureKind_E::Bp8_15},
        {"spectral_entropy_mag", FeatureKind_E::SpectralEntropyMag},

        // AR(4) magnitude
        {"ar4_c1",     FeatureKind_E::Ar4_c1},
        {"ar4_c2",     FeatureKind_E::Ar4_c2},
        {"ar4_c3",     FeatureKind_E::Ar4_c3},
        {"ar4_c4",     FeatureKind_E::Ar4_c4},
        {"ar4_resvar", FeatureKind_E::Ar4_resvar},

        // Custom
        {"tilt_variability",           FeatureKind_E::TiltVariability},
        {"vertical_acceleration_ratio",FeatureKind_E::VerticalAccelerationRatio},

        // Correlations
        {"xy_corr", FeatureKind_E::XyCorr},
        {"yz_corr", FeatureKind_E::YzCorr},
        {"zx_corr", FeatureKind_E::ZxCorr},

        // Extra magnitudes
        {"ptp_mag",           FeatureKind_E::PtpMag},
        {"zcr_mag",           FeatureKind_E::ZcrMag},
        {"dom_freq_mag",      FeatureKind_E::DomFreqMag},
        {"spec_centroid_mag", FeatureKind_E::SpecCentroidMag},
        {"spec_bandwidth_mag",FeatureKind_E::SpecBandwidthMag},
        {"spec_flatness_mag", FeatureKind_E::SpecFlatnessMag},
    };
} // anonymous namespace

// ==================== API HELPERS =============================================

// map from name to feature kind based on registry
static FeatureKind_E feature_kind_from_name(const std::string& name) {
    // iterate over registry looking for name
    for( auto it = g_feature_registry.begin() ; it != g_feature_registry.end() ; it++ ){
        if(it->first == name){
            // found
            return it->second;
        }
    }
    // not found -> throw error
    throw std::runtime_error("feature_kind_from_name: unknown feature name in JSON (not in registry)");
}

FeatureVector_C::FeatureVector_C(const OnnxConfigs_S& cfg) : cfg_(cfg), ctx_{}   {
    ops_.reserve(cfg_.feat_names.size());
    for (const auto& ftr : cfg_.feat_names){
        ops_.push_back(feature_kind_from_name(ftr));
    }
}

void FeatureVector_C::extract_xyz(const std::vector<accel_burst_t>& window,
        std::vector<float>& x,
        std::vector<float>& y,
        std::vector<float>& z)
{
    // reset
    x.clear();
    y.clear();
    z.clear();
    x.reserve(window.size());
    y.reserve(window.size());
    z.reserve(window.size());
    // every x within accel-burst-t gets appended to x vec
    // every y within accel-burst-t gets appended to y vec
    // every z within accel-burst-t gets appended to z vec
    accel_burst_t temp; 
    for(int i =0; i<window.size(); i++){
        x.push_back(window[i].x);
        y.push_back(window[i].y);
        z.push_back(window[i].z);
    }
}

// ========================== MAIN API ================================================
std::vector<float> FeatureVector_C::writeFeatureVector(const sliding_window_t& windowObj){
    // 0) get snapshot of ring buffer (data) part of window 
    std::vector<accel_burst_t> window;
    windowObj.sliding_window.get_data_snapshot(&window);
    // 1) extract channels for cache
    extract_xyz(window, ctx_.x, ctx_.y, ctx_.z);
    // 2) reset caches for this window
    ctx_.mag_computed_this_window = false;
    ctx_.psd_computed_this_window = false;
    ctx_.ar_computed_this_window  = false;
    ctx_.mag.clear();
    ctx_.freq.clear();
    ctx_.power.clear();
    ctx_.ar_resvar = 0.0f;
    for (int i = 0; i < AR_ORDER; ++i) {
        ctx_.ar_coeffs[i] = 0.0f;
    }

    // 3) compute ftrs in the order of cfg_.feat_names
    std::size_t num_ftrs = cfg_.feat_names.size();
    std::vector<float> ftrVec(num_ftrs);
    // index check
    if (ops_.size() != num_ftrs) {
        throw std::runtime_error("FeatureVector_C::writeFeatureVector: ops_ / feat_names size mismatch");
    }
    for(std::size_t i=0; i<num_ftrs; i++){
        FeatureKind_E currOp = ops_[i];
        ftrVec[i]=compute_one_feature(currOp, ctx_);
    }
    return ftrVec;
}


float FeatureVector_C::compute_one_feature(FeatureKind_E kind, FeatureCache_S& ctx)
{
    switch (kind) {
        // ---- Per-axis stats ----
        case FeatureKind_E::MeanAxisX: return mean_vec(ctx.x);
        case FeatureKind_E::MeanAxisY: return mean_vec(ctx.y);
        case FeatureKind_E::MeanAxisZ: return mean_vec(ctx.z);

        case FeatureKind_E::StdAxisX:  return std_vec(ctx.x);
        case FeatureKind_E::StdAxisY:  return std_vec(ctx.y);
        case FeatureKind_E::StdAxisZ:  return std_vec(ctx.z);

        case FeatureKind_E::RmsAxisX:  return rms_vec(ctx.x);
        case FeatureKind_E::RmsAxisY:  return rms_vec(ctx.y);
        case FeatureKind_E::RmsAxisZ:  return rms_vec(ctx.z);

        case FeatureKind_E::MadAxisX:  return mad_vec(ctx.x);
        case FeatureKind_E::MadAxisY:  return mad_vec(ctx.y);
        case FeatureKind_E::MadAxisZ:  return mad_vec(ctx.z);

        case FeatureKind_E::IqrAxisX:  return iqr_vec(ctx.x);
        case FeatureKind_E::IqrAxisY:  return iqr_vec(ctx.y);
        case FeatureKind_E::IqrAxisZ:  return iqr_vec(ctx.z);

        // ---- Magnitude / time-domain ----
        case FeatureKind_E::SmaXYZ:    return sma_xyz(ctx);
        case FeatureKind_E::MeanMag:   return mean_mag(ctx);
        case FeatureKind_E::StdMag:    return std_mag(ctx);
        case FeatureKind_E::RmsMag:    return rms_mag(ctx);
        case FeatureKind_E::MadMag:    return mad_mag(ctx);
        case FeatureKind_E::IqrMag:    return iqr_mag(ctx);

        // ---- PSD / bandpower / entropy ----
        case FeatureKind_E::TotalPowerMag:
            return total_power_mag(ctx);
        case FeatureKind_E::Bp0p5_3:
            return bandpower_mag(ctx, 0.5f, 3.0f);
        case FeatureKind_E::Bp3_8:
            return bandpower_mag(ctx, 3.0f, 8.0f);
        case FeatureKind_E::Bp8_15:
            return bandpower_mag(ctx, 8.0f, 15.0f);
        case FeatureKind_E::SpectralEntropyMag:
            return spectral_entropy_mag(ctx);

        // ---- AR(4) magnitude ----
        case FeatureKind_E::Ar4_c1: {
            Ar4Result ar = ar4_mag(ctx);
            return ar.c1;
        }
        case FeatureKind_E::Ar4_c2: {
            Ar4Result ar = ar4_mag(ctx);
            return ar.c2;
        }
        case FeatureKind_E::Ar4_c3: {
            Ar4Result ar = ar4_mag(ctx);
            return ar.c3;
        }
        case FeatureKind_E::Ar4_c4: {
            Ar4Result ar = ar4_mag(ctx);
            return ar.c4;
        }
        case FeatureKind_E::Ar4_resvar: {
            Ar4Result ar = ar4_mag(ctx);
            return ar.resvar;
        }

        // ---- Custom motion features ----
        case FeatureKind_E::TiltVariability: {
            const auto& x = ctx.x;
            const auto& y = ctx.y;
            const auto& z = ctx.z;
            size_t n = std::min({x.size(), y.size(), z.size()});
            if (n == 0) return 0.0f;
            std::vector<float> tilt(n);
            for (size_t i = 0; i < n; ++i) {
                float horiz = std::sqrt(x[i] * x[i] + y[i] * y[i]);
                tilt[i] = std::atan2(z[i], horiz);
            }
            return std_vec(tilt);
        }

        case FeatureKind_E::VerticalAccelerationRatio: {
            float sz = std_vec(ctx.z);
            float sx = std_vec(ctx.x);
            float sy = std_vec(ctx.y);
            return sz / (sx + sy + 1e-6f);
        }

        // ---- Correlations between axes ----
        case FeatureKind_E::XyCorr: return corr_vec(ctx.x, ctx.y);
        case FeatureKind_E::YzCorr: return corr_vec(ctx.y, ctx.z);
        case FeatureKind_E::ZxCorr: return corr_vec(ctx.z, ctx.x);

        // ---- Extra magnitude features ----
        case FeatureKind_E::PtpMag:           return ptp_mag(ctx);
        case FeatureKind_E::ZcrMag: {
            // zero-crossing on mag after DC removal
            const auto& m = mag_from_xyz(ctx);
            if (m.size() < 2) return 0.0f;
            float mean_m = mean_vec(m);
            int crossings = 0;
            float prev = m[0] - mean_m;
            for (size_t i = 1; i < m.size(); ++i) {
                float cur = m[i] - mean_m;
                if ((prev > 0 && cur < 0) || (prev < 0 && cur > 0)) {
                    ++crossings;
                }
                prev = cur;
            }
            return static_cast<float>(crossings) /
                   static_cast<float>(m.size() - 1);
        }
        case FeatureKind_E::DomFreqMag:       return dom_freq_mag(ctx);
        case FeatureKind_E::SpecCentroidMag:  return spec_centroid_mag(ctx);
        case FeatureKind_E::SpecBandwidthMag: return spec_bandwidth_mag(ctx);
        case FeatureKind_E::SpecFlatnessMag:  return spec_flatness_mag(ctx);

        case FeatureKind_E::Unknown:
        default:
            throw std::runtime_error("compute_one_feature: Unknown FeatureKind");
    }
}

// ============================ STAT HELPERS ===================================

static float mean_vec(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    float sum = std::accumulate(v.begin(), v.end(), 0.0f);
    return sum / static_cast<float>(v.size());
}

static float std_vec(const std::vector<float>& v) {
    if (v.size() < 2) return 0.0f;
    float m = mean_vec(v);
    float acc = 0.0f;
    for (float x : v) {
        float d = x - m;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<float>(v.size()));
}

static float rms_vec(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    float acc = 0.0f;
    for (float x : v) acc += x * x;
    return std::sqrt(acc / static_cast<float>(v.size()));
}

static float median_vec(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    const size_t mid = n / 2;
    if (n % 2 == 1) {
        // odd
        return v[mid];
    } else {
        // even: average of the two center values
        return 0.5f * (v[mid - 1] + v[mid]);
    }
}

static float mad_vec(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;

    // 1) median of original values
    std::vector<float> tmp = v;
    float m = median_vec(tmp);

    // 2) abs deviations
    for (float& x : tmp) {
        x = std::fabs(x - m);
    }

    // 3) median of abs deviations
    return median_vec(tmp);
}


static float iqr_vec(std::vector<float> v) {
    if (v.size() < 2) return 0.0f;
    std::sort(v.begin(), v.end());
    auto q = [&](float qv) {
        float pos = qv * (v.size() - 1);
        size_t idx = static_cast<size_t>(pos);
        float frac = pos - idx;
        if (idx + 1 < v.size())
            return v[idx] * (1.0f - frac) + v[idx + 1] * frac;
        return v[idx];
    };
    float q25 = q(0.25f);
    float q75 = q(0.75f);
    return q75 - q25;
}

static float corr_vec(const std::vector<float>& a,
                      const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n < 2) return 0.0f;

    float ma = mean_vec(a);
    float mb = mean_vec(b);
    float num = 0.0f, va = 0.0f, vb = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float da = a[i] - ma;
        float db = b[i] - mb;
        num += da * db;
        va  += da * da;
        vb  += db * db;
    }
    if (va <= 0.0f || vb <= 0.0f) return 0.0f;
    return num / std::sqrt(va * vb);
}

// ============================ MAG + PSD + AR ================================

// magnitude cache
static const std::vector<float>& mag_from_xyz(FeatureCache_S& ctx) {
    if (!ctx.mag_computed_this_window) {
        const auto& x = ctx.x;
        const auto& y = ctx.y;
        const auto& z = ctx.z;
        size_t n = std::min({x.size(), y.size(), z.size()});
        ctx.mag.resize(n);
        for (size_t i = 0; i < n; ++i) {
            float xi = x[i];
            float yi = y[i];
            float zi = z[i];
            ctx.mag[i] = std::sqrt(xi * xi + yi * yi + zi * zi);
        }
        ctx.mag_computed_this_window = true;
    }
    return ctx.mag;
}

// Naive DFT-based PSD (fine for your N); swap for FFT later if you want
static void compute_psd_if_needed(FeatureCache_S& ctx) {
    if (ctx.psd_computed_this_window) return;

    const auto& m = mag_from_xyz(ctx);
    size_t N = m.size();
    if (N == 0) {
        ctx.freq.clear();
        ctx.power.clear();
        ctx.psd_computed_this_window = true;
        return;
    }

    size_t K = N / 2 + 1;  // one-sided
    ctx.freq.resize(K);
    ctx.power.resize(K);

    const float fs = ctx.fs;

    for (size_t k = 0; k < K; ++k) {
        double re = 0.0;
        double im = 0.0;
        double omega = -2.0 * 3.14 * static_cast<double>(k) / static_cast<double>(N);
        for (size_t n = 0; n < N; ++n) {
            double angle = omega * static_cast<double>(n);
            double val = m[n];
            re += val * std::cos(angle);
            im += val * std::sin(angle);
        }
        double mag2 = (re * re + im * im) / static_cast<double>(N);
        ctx.freq[k]  = (fs * static_cast<float>(k)) / static_cast<float>(N);
        ctx.power[k] = static_cast<float>(mag2);
    }

    ctx.psd_computed_this_window = true;
}

static float bandpower_mag(FeatureCache_S& ctx, float lo, float hi) {
    compute_psd_if_needed(ctx);
    if (ctx.freq.empty()) return 0.0f;

    float acc = 0.0f;
    for (size_t k = 0; k < ctx.freq.size(); ++k) {
        float f = ctx.freq[k];
        if (f >= lo && f <= hi) {
            acc += ctx.power[k];
        }
    }
    return acc;
}

static float total_power_mag(FeatureCache_S& ctx) {
    compute_psd_if_needed(ctx);
    float acc = 0.0f;
    for (float p : ctx.power) acc += p;
    return acc;
}

static float spectral_entropy_mag(FeatureCache_S& ctx) {
    compute_psd_if_needed(ctx);
    const auto& P = ctx.power;
    if (P.empty()) return 0.0f;

    double sumP = 0.0;
    for (float p : P) {
        sumP += std::max(static_cast<double>(p), 0.0);
    }
    if (sumP <= 0.0) return 0.0f;

    double H_nat = 0.0;
    for (float p : P) {
        double pk = std::max(static_cast<double>(p), 0.0) / sumP;
        if (pk > 0.0) {
            H_nat += -pk * std::log(pk);  // natural log
        }
    }
    // convert nats -> bits
    const double LOG2 = std::log(2.0);
    double H_bits = H_nat / LOG2;
    return static_cast<float>(H_bits);
}

static float dom_freq_mag(FeatureCache_S& ctx) {
    compute_psd_if_needed(ctx);
    if (ctx.freq.empty()) return 0.0f;
    auto it = std::max_element(ctx.power.begin(), ctx.power.end());
    size_t idx = std::distance(ctx.power.begin(), it);
    return ctx.freq[idx];
}

static float spec_centroid_mag(FeatureCache_S& ctx) {
    compute_psd_if_needed(ctx);
    const auto& f = ctx.freq;
    const auto& P = ctx.power;
    if (f.empty() || P.empty()) return 0.0f;

    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < f.size(); ++i) {
        num += static_cast<double>(f[i]) * static_cast<double>(P[i]);
        den += static_cast<double>(P[i]);
    }
    if (den <= 0.0) return 0.0f;
    return static_cast<float>(num / den);
}

static float spec_bandwidth_mag(FeatureCache_S& ctx) {
    float centroid = spec_centroid_mag(ctx);
    compute_psd_if_needed(ctx);
    const auto& f = ctx.freq;
    const auto& P = ctx.power;
    if (f.empty() || P.empty()) return 0.0f;

    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < f.size(); ++i) {
        double df = static_cast<double>(f[i]) - centroid;
        num += static_cast<double>(P[i]) * df * df;
        den += static_cast<double>(P[i]);
    }
    if (den <= 0.0) return 0.0f;
    return static_cast<float>(std::sqrt(num / den));
}

static float spec_flatness_mag(FeatureCache_S& ctx) {
    compute_psd_if_needed(ctx);
    const auto& P = ctx.power;
    if (P.empty()) return 0.0f;

    const float eps = 1e-12f;
    double log_sum = 0.0;
    double lin_sum = 0.0;
    for (float p : P) {
        float v = p + eps;
        log_sum += std::log(v);
        lin_sum += v;
    }
    double K = static_cast<double>(P.size());
    double geo_mean = std::exp(log_sum / K);
    double ar_mean  = lin_sum / K;
    if (ar_mean <= 0.0) return 0.0f;
    return static_cast<float>(geo_mean / ar_mean);
}

// AR(4)
static void compute_ar4_if_needed(FeatureCache_S& ctx) {
    if (ctx.ar_computed_this_window) return;

    const auto& m_in = mag_from_xyz(ctx);
    const size_t N = m_in.size();
    const int p = AR_ORDER;

    // defaults
    for (int i = 0; i < p; ++i) ctx.ar_coeffs[i] = 0.0f;
    ctx.ar_resvar = 0.0f;

    if (N <= static_cast<size_t>(p + 2)) {
        ctx.ar_computed_this_window = true;
        return;
    }

    // 1) copy to double + demean
    std::vector<double> m(N);
    double mean = 0.0;
    for (size_t i = 0; i < N; ++i) {
        mean += static_cast<double>(m_in[i]);
    }
    mean /= static_cast<double>(N);
    for (size_t i = 0; i < N; ++i) {
        m[i] = static_cast<double>(m_in[i]) - mean;
    }

    const size_t M = N - p;  // number of rows in X, length of Y

    // 2) Build normal equations: XtX (p x p), Xty (p)
    double XtX[4][4] = {};
    double Xty[4]    = {};

    for (size_t t = 0; t < M; ++t) {
        double y = m[p + t];

        // x_j = m[p - j - 1 + t], j=0..p-1
        double x[4];
        for (int j = 0; j < p; ++j) {
            size_t idx = static_cast<size_t>(p - j - 1) + t;
            x[j] = m[idx];
        }

        // accumulate XᵀX and Xᵀy
        for (int j = 0; j < p; ++j) {
            Xty[j] += x[j] * y;
            for (int k = 0; k < p; ++k) {
                XtX[j][k] += x[j] * x[k];
            }
        }
    }

    // 3) Solve XtX * a = Xty (simple Gaussian elimination, p=4)
    double A[4][5]; // augmented matrix [XtX | Xty]
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            A[i][j] = XtX[i][j];
        }
        A[i][p] = Xty[i];
    }

    // Gaussian elimination
    for (int i = 0; i < p; ++i) {
        // pivot
        double pivot = A[i][i];
        if (std::fabs(pivot) < 1e-12) {
            // singular-ish; leave coefficients at 0
            ctx.ar_computed_this_window = true;
            return;
        }
        // normalize row
        for (int j = i; j <= p; ++j) {
            A[i][j] /= pivot;
        }
        // eliminate
        for (int r = 0; r < p; ++r) {
            if (r == i) continue;
            double factor = A[r][i];
            for (int c = i; c <= p; ++c) {
                A[r][c] -= factor * A[i][c];
            }
        }
    }

    double a[4];
    for (int i = 0; i < p; ++i) {
        a[i] = A[i][p];
    }

    // 4) Residual variance: np.var(resid)
    double acc_resid2 = 0.0;
    for (size_t t = 0; t < M; ++t) {
        double y = m[p + t];
        double y_hat = 0.0;
        for (int j = 0; j < p; ++j) {
            size_t idx = static_cast<size_t>(p - j - 1) + t;
            y_hat += a[j] * m[idx];
        }
        double r = y - y_hat;
        acc_resid2 += r * r;
    }
    double var = acc_resid2 / static_cast<double>(M);  // population variance

    for (int i = 0; i < p; ++i) {
        ctx.ar_coeffs[i] = static_cast<float>(a[i]);
    }
    ctx.ar_resvar = static_cast<float>(var);
    ctx.ar_computed_this_window = true;
}


static Ar4Result ar4_mag(FeatureCache_S& ctx) {
    compute_ar4_if_needed(ctx);
    Ar4Result out;
    out.c1     = ctx.ar_coeffs[0];
    out.c2     = ctx.ar_coeffs[1];
    out.c3     = ctx.ar_coeffs[2];
    out.c4     = ctx.ar_coeffs[3];
    out.resvar = ctx.ar_resvar;
    return out;
}

// ---- time-domain / mag features using cache ----
static float sma_xyz(FeatureCache_S& ctx) {
    size_t n = std::min({ctx.x.size(), ctx.y.size(), ctx.z.size()});
    if (n == 0) return 0.0f;
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        acc += std::fabs(ctx.x[i]) + std::fabs(ctx.y[i]) + std::fabs(ctx.z[i]);
    }
    return acc / static_cast<float>(n);
}

static float mean_mag(FeatureCache_S& ctx) {
    return mean_vec(mag_from_xyz(ctx));
}

static float std_mag(FeatureCache_S& ctx) {
    return std_vec(mag_from_xyz(ctx));
}

static float rms_mag(FeatureCache_S& ctx) {
    return rms_vec(mag_from_xyz(ctx));
}

static float mad_mag(FeatureCache_S& ctx) {
    return mad_vec(mag_from_xyz(ctx));
}

static float iqr_mag(FeatureCache_S& ctx) {
    return iqr_vec(mag_from_xyz(ctx));
}

static float ptp_mag(FeatureCache_S& ctx) {
    const auto& m = mag_from_xyz(ctx);
    if (m.empty()) return 0.0f;
    auto [mn_it, mx_it] = std::minmax_element(m.begin(), m.end());
    return *mx_it - *mn_it;
}