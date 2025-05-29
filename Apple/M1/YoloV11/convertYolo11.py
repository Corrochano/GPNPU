"""
Copyright 2024 Álvaro Corrochano López

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import coremltools as ct
import torch
from ultralytics import YOLO

# Load the model
source_model = YOLO('yolo11x.pt')

source_model.export(format="torchscript", task="detect")  # creates 'yolo11n.torchscript'

# Load the exported TorchScript model
torchscript_model = torch.jit.load("yolo11x.torchscript")

# Convert to CoreML
dummy_input = torch.rand(1, 3, 640, 640)

fp16model = ct.convert(torchscript_model, 
                   convert_to="mlprogram",
                   inputs=[ct.TensorType(shape=dummy_input.shape)],
                   source="pytorch", 
                   compute_precision=ct.precision.FLOAT16)
                   
fp32model = ct.convert(torchscript_model, 
                   convert_to="mlprogram",
                   inputs=[ct.TensorType(shape=dummy_input.shape)],
                   source="pytorch", 
                   compute_precision=ct.precision.FLOAT32)
                   
# Save the models                   
fp16model.save("yolo11xFP16.mlpackage")

fp32model.save("yolo11xFP32.mlpackage")


