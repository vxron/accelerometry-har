#pragma once
#include "ONXXClassifier.hpp"
#include "window_configs.hpp"
#include <string>
#include <stdexcept>

constexpr int AR_ORDER = 4;

enum class FeatureKind_E {
    // Per-axis stats
    MeanAxisX,
    MeanAxisY,
    MeanAxisZ,
    StdAxisX,
    StdAxisY,
    StdAxisZ,
    RmsAxisX,
    RmsAxisY,
    RmsAxisZ,
    MadAxisX,
    MadAxisY,
    MadAxisZ,
    IqrAxisX,
    IqrAxisY,
    IqrAxisZ,
    // Magnitude/time-domain features
    SmaXYZ,
    MeanMag,
    StdMag,
    RmsMag,
    MadMag,
    IqrMag,
    // PSD / bandpower / entropy
    TotalPowerMag,
    Bp0p5_3,
    Bp3_8,
    Bp8_15,
    SpectralEntropyMag,
    // AR(4) magnitude
    Ar4_c1,
    Ar4_c2,
    Ar4_c3,
    Ar4_c4,
    Ar4_resvar,
    // Custom motion features
    TiltVariability,
    VerticalAccelerationRatio,
    // Correlations between axes
    XyCorr,
    YzCorr,
    ZxCorr,
    // Extra magnitude features
    PtpMag,
    ZcrMag,
    DomFreqMag,
    SpecCentroidMag,
    SpecBandwidthMag,
    SpecFlatnessMag,
    // Error catching
    Unknown,
};

// Cache for expensive features so we dont reconstruct objects every window (fourier/AR)
struct FeatureCache_S {
    // individual axis datastreams cached once per window
    const std::vector<float>& x;
    const std::vector<float>& y;
    const std::vector<float>& z;
    std::size_t fs = FS_HZ; // sampling freq

    // large ftr cache
    bool mag_computed_this_window = false;
    std::vector<float> mag;
    bool psd_computed_this_window = false;
    std::vector<float> freq, power;
    
    bool ar_computed_this_window = false;
    float ar_coeffs[AR_ORDER];
    float ar_resvar;
}

class FeatureVector_C {
public: // public API
    // featurevector computing requires specific model configs with it
    explicit FeatureVector_C(const OnnxConfigs_S& cfg);
    // will write per window
    std::vector<float> writeFeatureVector(sliding_window_t window); 
    // ^should get a copy of window (so pass by value)
private:
    const OnnxConfigs_S& cfg_; // ref to classifier meta
    FeatureCache_S currCtx_; 
    std::vector<FeatureKind_E> ops_; // list of operating enum ftr kinds we must get for these cfgs (init on construction)
    computeAllFeatures(OnnxConfigs_S cfgs); // will use several static methods in implementation
    // helper to get x/y/z from sliding_window_t composed of accel_burst_t
    static float compute_one_feature(FeatureKind_E kind, FeatureCache_S& ctx);
    static void extract_xyz(sliding_window_t window,
        std::vector<float>& x,
        std::vector<float>& y,
        std::vector<float>& z);
};


// can be called like
// writeFeatureVector(&slidingWindow.feature_vector, onxxClassifier.cfgs_)