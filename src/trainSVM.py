"""
This file must:
(1) Train Kernel SVM
- Load data
- Extract features
- Find best kernel function by testing results
- Apply optimal kernel and C-value
(2) Export to ONNX model
- Zipmap disabled (so C++ gets a plain tensor)
"""

# Training order will be walk -> stand -> sit -> jump
# So we need to go through data and when theres a sequence of ones, add column for walk, and next time u run into ones, u know its stand, etc



