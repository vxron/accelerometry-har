#pragma once
#include "ONXXClassifier.hpp"

class FeatureExtractor_C {
public: // public API
    // requires copy of onxx classifier meta cfgs
    writeFeatureVector(std::vector<float>* dest, OnnxConfigs_S cfgs);
private:
    computeAllFeatures(OnnxConfigs_S cfgs); // will use several static methods in implementation
    // uses feat names to get them (much match json)
};

// can be called like
// writeFeatureVector(&slidingWindow.feature_vector, onxxClassifier.cfgs_)