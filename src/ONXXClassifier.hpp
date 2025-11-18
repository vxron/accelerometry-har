// HarOnnxClassifier.hpp
/*
Goal: 
*/
#pragma once
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

// These must come from parsing json meta.
struct OnnxConfigs_S {
    std::vector<std::string> feat_names; // global order for full feature vector
    // indices for features we've selected for diff classifiers
    std::vector<int> intensity_idx;
    std::vector<int> static_idx;
    std::vector<int> dynamic_idx;
    // per branch labels (intensity preclassifier, svm1, svm2)
    int intensity_static;
    int intensity_dynamic;
    int static_sit;
    int static_stand;
    int dynamic_walk;
    int dynamic_turn;
    // labels to map back to class from classifier output
    int stand_id;
    int sit_id;
    int walk_id;
    int turn_id;
};

class OnnxClassifier_C {
public:
    explicit OnnxClassifier_C(const std::string& meta_json_path); // parse configs upon construction
    // Simple PUBLIC API for making classifications from input vec
    // full_features should contain all ftrs used in one of the 3 classifiers and match feat-names
    int classify(const std::vector<float>& full_features);

    // config getter for feature extractor and other modules who need to know (& cant change these configs)
    const OnnxConfigs_S getConfigs() const { return cfg_; };
private:
    int run_model(const std::vector<float>& full_features, ClassifierStages_E model) const;
    
    // ONXX runtime stuff
    Ort::Env env_;
    Ort::Session sess_intensity_;
    Ort::Session sess_static_;
    Ort::Session sess_dynamic_;
    Ort::MemoryInfo mem_info_;

    // Cached IO names (so we don't reallocate on the heap each time we classify)
    std::string input_name_int_;
    std::string input_name_static_;
    std::string input_name_dynamic_;

    std::string output_name_int_;
    std::string output_name_static_;
    std::string output_name_dynamic_;

    OnnxConfigs_S cfg_;
};
