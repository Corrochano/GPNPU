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
import argparse
import coremltools as ct
import torch
from ultralytics import YOLO

def main(model):
    name = model.split('.')[0]
    # Load the model
    source_model = YOLO(model) #'yolo11x.pt')

    source_model.export(format="torchscript", task="detect")  # creates 'yolo11n.torchscript'

    # Load the exported TorchScript model
    torchscript_model = torch.jit.load(model)

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
    fp16model.save(f"{name}FP16.mlpackage")

    fp32model.save(f"{name}FP32.mlpackage")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="python convertYolo11.py model_name")
    
    # Add arguments
    parser.add_argument("model_name", type=str, help="the name of the model to convert")
    
    # Parse the arguments
    args = parser.parse_args()

    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.model_name is None :
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Call the main function with provided arguments
        main(args.model_name)    

