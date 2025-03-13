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
import torch
import coremltools as ct

def main(device):
    print("[INFO] Warming up...")

    print("[INFO] Creating input grid...")
    # Create random input grid
    x = torch.linspace(0, 1, 1000, dtype=torch.float16)
    y = torch.linspace(0, 1, 1000, dtype=torch.float16)
    X, Y = torch.meshgrid(x, y)
    
    x = torch.exp(torch.mul( # If I cast everything, there are still operations on INT32 idk why and the performance and consume increase a lot
                    -50, 
                    torch.add(torch.pow((X - 0.5), 2), torch.pow((Y - 0.5), 2))
                ))
    x = x.unsqueeze(0).unsqueeze(0) # Channel and batch size. Necessary for conv layer
    x_prev = x.clone()

    mask = torch.ones_like(x)
    mask[:, :, 0, :] = 0        # Top boundary
    mask[:, :, -1, :] = 0       # Bottom boundary
    mask[:, :, :, 0] = 0        # Left boundary
    mask[:, :, :, -1] = 0       # Right boundary       

    # Prepare inputs for the model
    warmup_dict = {'X': x, 'X_prev': x_prev, 'Mask': mask}

    print("[INFO] Loading warmup model...")
    if device == "ane":
        wu_model = ct.models.MLModel('jacobi1000_model_fp16_100.mlpackage', compute_units=ct.ComputeUnit.CPU_AND_NE)
    elif device == "gpu":
        wu_model = ct.models.MLModel('jacobi1000_model_fp16_100.mlpackage', compute_units=ct.ComputeUnit.CPU_AND_GPU)
    elif device == "cpu":
        wu_model = ct.models.MLModel('jacobi1000_model_fp16_100.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY)
    elif device == "all":
        wu_model = ct.models.MLModel('jacobi1000_model_fp16_100.mlpackage', compute_units=ct.ComputeUnit.ALL)

    print("[INFO] Running warmup...")
    for i in range(100):
        wu_model.predict(warmup_dict)

    print("+++ Warmup Done +++")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="python warmup.py device")

    # Add arguments
    parser.add_argument("device_input", type=str, help="device: cpu/gpu/ane/all")

    # Parse the arguments
    args = parser.parse_args()

    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.device_input is None:
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Call the main function with provided arguments
        main(args.device_input)
        
