#include "ONXXClassifier.hpp"
#include <fstream>
#include "json.hpp"
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include "logger.hpp"

using json = nlohmann::json;

OnnxClassifier_C::OnnxClassifier_C(const std::string& meta_json_path) 
  : env_(ORT_LOGGING_LEVEL_WARNING, "har"), mem_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) 
{
    // ====== (1) load it ===========
    std::ifstream json_meta(meta_json_path);
    if (!json_meta) {
        throw std::runtime_error("Failed to open meta JSON: " + meta_json_path);
    }
    LOG_ALWAYS("OnnxClassifier: meta JSON path = " << meta_json_path);
    json metadata = json::parse(json_meta);
    
    // ========= (2) parse it ===========
    // directory for ONNX files (same folder as meta)
    std::filesystem::path meta_path(meta_json_path);
    std::filesystem::path base_dir = meta_path.parent_path();
    
    
    cfg_.feat_names = metadata.at("feat_names").get<std::vector<std::string>>();
    
    // dynamic
    const auto& meta_dyn = metadata.at("dynamic_branch");
    cfg_.dynamic_idx = meta_dyn.at("feature_indices").get<std::vector<int>>();
    const auto& branch_labels = meta_dyn.at("branch_labels");
    cfg_.dynamic_turn = branch_labels.at("turn").get<int>();
    cfg_.dynamic_walk = branch_labels.at("walk").get<int>();
    
    // preclassifier
    const auto& meta_int = metadata.at("intensity");
    cfg_.intensity_idx = meta_int.at("feature_indices").get<std::vector<int>>();
    const auto& branch1_labels = meta_int.at("classes");
    cfg_.intensity_static = branch1_labels.at("static").get<int>();;
    cfg_.intensity_dynamic = branch1_labels.at("dynamic").get<int>();;
    
    // static
    const auto& meta_static = metadata.at("static_branch");
    cfg_.static_idx = meta_static.at("feature_indices").get<std::vector<int>>();
    const auto& branch2_labels = meta_static.at("branch_labels");
    cfg_.static_sit = branch2_labels.at("sit").get<int>();
    cfg_.static_stand = branch2_labels.at("stand").get<int>();

    // build all the model paths
    std::string dyn_onnx_file = meta_dyn.at("onnx_file").get<std::string>();
    std::string int_onnx_file = meta_int.at("onnx_file").get<std::string>();
    std::string static_onnx_file = meta_static.at("onnx_file").get<std::string>();
    
    // global class labels
    const auto& label_to_id = metadata.at("label_name_to_id");
    cfg_.stand_id = label_to_id.at("stand").get<int>();
    cfg_.sit_id = label_to_id.at("sit").get<int>();
    cfg_.walk_id = label_to_id.at("walk").get<int>();
    cfg_.turn_id = label_to_id.at("turn").get<int>();

    // ========= (3) create ONNX sessions (models) =============
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1); // Pi is small, needs to be lightweight
    opts.SetGraphOptimizationLevel(
        ORT_ENABLE_BASIC
    );

    auto int_model_path    = (base_dir / int_onnx_file).string();
    auto static_model_path = (base_dir / static_onnx_file).string();
    auto dyn_model_path    = (base_dir / dyn_onnx_file).string();

#ifdef _WIN32
    std::wstring w_int    = std::filesystem::path(int_model_path).wstring();
    std::wstring w_static = std::filesystem::path(static_model_path).wstring();
    std::wstring w_dyn    = std::filesystem::path(dyn_model_path).wstring();

    sess_intensity_ = std::make_unique<Ort::Session>(env_, w_int.c_str(), opts);
    sess_static_    = std::make_unique<Ort::Session>(env_, w_static.c_str(), opts);
    sess_dynamic_   = std::make_unique<Ort::Session>(env_, w_dyn.c_str(), opts);
#else
    sess_intensity_ = std::make_unique<Ort::Session>(env_, int_model_path.c_str(),    opts);
    sess_static_    = std::make_unique<Ort::Session>(env_, static_model_path.c_str(), opts);
    sess_dynamic_   = std::make_unique<Ort::Session>(env_, dyn_model_path.c_str(),    opts);
#endif

    // ========== (4) cache input/output tensor names ==============
    // (this is bcuz onnx runtime stores io names dynamically as heap allocated strings & we don't want to reallocate everytime we make a classification)
    Ort::AllocatorWithDefaultOptions allocator;

    {
        auto in_name  = sess_intensity_->GetInputNameAllocated(0, allocator);
        auto out_name = sess_intensity_->GetOutputNameAllocated(0, allocator);
        input_name_int_  = in_name.get();
        output_name_int_ = out_name.get();
    }
    {
        auto in_name  = sess_static_->GetInputNameAllocated(0, allocator);
        auto out_name = sess_static_->GetOutputNameAllocated(0, allocator);
        input_name_static_  = in_name.get();
        output_name_static_ = out_name.get();
    }
    {
        auto in_name  = sess_dynamic_->GetInputNameAllocated(0, allocator);
        auto out_name = sess_dynamic_->GetOutputNameAllocated(0, allocator);
        input_name_dynamic_  = in_name.get();
        output_name_dynamic_ = out_name.get();
    }

}

int OnnxClassifier_C::run_model(const std::vector<float>& full_features, ClassifierStage_E model) const {
    // ====== (1) gather ptrs to what we need for the specific model ======
    const std::vector<int>* ftr_idxs = nullptr; // features selected for it
    Ort::Session* sess = nullptr; // ref to onxx model
    const char* input_name = nullptr; // input tensor allocated name
    const char* output_name = nullptr; // output tensor allocated name

    switch(model){
        case ClassifierStage_E::Intensity:
            ftr_idxs    = &cfg_.intensity_idx;
            sess        = sess_intensity_.get();
            input_name  = input_name_int_.c_str(); // cached i/o, cstr is alr a ptr
            output_name = output_name_int_.c_str();
            break;
        case ClassifierStage_E::StaticBranch:
            ftr_idxs    = &cfg_.static_idx;
            sess        = sess_static_.get();
            input_name  = input_name_static_.c_str();
            output_name = output_name_static_.c_str();
            break;
        case ClassifierStage_E::DynamicBranch:
            ftr_idxs    = &cfg_.dynamic_idx;
            sess        = sess_dynamic_.get();
            input_name  = input_name_dynamic_.c_str();
            output_name = output_name_dynamic_.c_str();
            break;
        default:
            break;
    }

    if (!ftr_idxs || !sess || !input_name || !output_name) {
        throw std::runtime_error("run_model: classifier configuration not initialized properly.");
    }

    // ======= (2) build subset of ftrs from full set in the order expected by onnx ========
    std::vector<float> input_vals;
    input_vals.resize(ftr_idxs->size()); // avoid reallocations on push_backs
    // iterate over copies of feature elements
    std::size_t currIdx = 0;
    for(auto ftr : full_features){
        auto it = std::find(ftr_idxs->begin(), ftr_idxs->end(),currIdx);
        if (it!=ftr_idxs->end()){ // there was an occurence found; choose this ftr
            // get idx
            std::size_t pos = static_cast<std::size_t>(std::distance(ftr_idxs->begin(), it));
            // safety guard
            if (pos >= input_vals.size()) {
                throw std::out_of_range("run_model: feature position out of range.");
            }
            input_vals[pos]=ftr; // place ftr in correct onnx order
        }
        currIdx++;
    }
    
    // ======== (3) create onnx input tensor shape [1, n_features] ===========
    int64_t dims[2] = {1, static_cast<int64_t>(input_vals.size())};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,                                 // CPU memory info
        input_vals.data(),                         // ptr to contig ftrs
        static_cast<size_t>(input_vals.size()),    // num elements (ftrs)
        dims,                                      // shape [1,n_ftrs]  
        2                                          // rank (2dim tensor, so a matrix) 
    );

    // ======= (4) run the session ==============
    const char* inputName[]  = { input_name };
    const char* outputName[] = { output_name };

    auto output_tensors = sess->Run(
        Ort::RunOptions{nullptr},
        inputName,
        &input_tensor,
        1,
        outputName,
        1
    );

    if (output_tensors.size() != 1) {
        throw std::runtime_error("run_model: expected exactly one output tensor.");
    }

    Ort::Value& out = output_tensors[0];

    // Assume the model outputs a single scalar class id (int64)
    int64_t* out_data = out.GetTensorMutableData<int64_t>();
    // If the pipeline outputs [N,1] still get a contiguous buffer; just with N=1 (still need to index)
    int result = static_cast<int>(out_data[0]);
    return result;   // 0 or 1

}

int OnnxClassifier_C::classify(const std::vector<float>& full_features) {
    
    int intensity_class = run_model(full_features, ClassifierStage_E::Intensity);
    if(intensity_class == cfg_.intensity_static) {
        int static_class = run_model(full_features, ClassifierStage_E::StaticBranch);
        if(static_class == cfg_.static_sit){
            return cfg_.sit_id;
        }
        else if(static_class == cfg_.static_stand){
            return cfg_.stand_id;
        }
        else {
            throw std::runtime_error("Failed at level 2 - Static SVM Returned Invalid");
        }
    }
    else if(intensity_class == cfg_.intensity_dynamic) {
        int dyn_class = run_model(full_features, ClassifierStage_E::DynamicBranch);
        if(dyn_class == cfg_.dynamic_walk){
            return cfg_.walk_id;
        }
        else if (dyn_class == cfg_.dynamic_turn){
            return cfg_.turn_id;
        }
        else {
            throw std::runtime_error("Failed at level 2 - Dynamic SVM Returned Invalid");
        }
    }
    else {
        throw std::runtime_error("Failed at level 1 - Intensity Preclassifier Returned Invalid");
    }

    return -1; 
}