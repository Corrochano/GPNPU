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
from torch import nn
import coremltools as ct

def main(device):
    print("[INFO] Warming up...")

    print("[INFO] Creating input grid...")
    # Create random input grid
    nx=1024
    ny=1024
    dt=0.001
    alpha=0.01 

    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)  # spatial step sizes

    x = torch.linspace(0, 1, nx, dtype=torch.float16)
    y = torch.linspace(0, 1, ny, dtype=torch.float16)
    X, Y = torch.meshgrid(x, y)
    
    x = torch.exp(torch.mul( # Take the value of X
                        -50, 
                        torch.add(torch.pow((X - 0.5), 2), torch.pow((Y - 0.5), 2))
                    )).to(torch.float16)
    x = x.unsqueeze(0).unsqueeze(0)    
    
    # Create masks
    mask = torch.ones_like(x, dtype=torch.float16)
    mask[:, :, 0, :] = 0
    mask[:, :, -1, :] = 0
    mask[:, :, :, 0] = 0
    mask[:, :, :, -1] = 0    

    # Define num_levels
    num_levels = 9
    
    masks = [mask]
    
    for _ in range(num_levels):# precalculate masks
        masks.append(nn.AvgPool2d(kernel_size=2)(masks[-1]).to(torch.float16))    
   
    # Prepare inputs for the model
    warmup_dict = {'X': X, 'Y': Y, 'Mask1': masks[0], 'Mask2': masks[1], 'Mask3': masks[2], 'Mask4': masks[3], 'Mask5': masks[4], 'Mask6': masks[5], 'Mask7': masks[6], 'Mask8': masks[7], 'Mask9': masks[8],
    'Mask10': masks[9]}

    print("[INFO] Loading warmup model...")
    if device == "ane":
        wu_model = ct.models.MLModel('jacobi1024_model_fp16_100.mlpackage', compute_units=ct.ComputeUnit.CPU_AND_NE)
    elif device == "gpu":
        wu_model = ct.models.MLModel('jacobi1024_model_fp16_100.mlpackage', compute_units=ct.ComputeUnit.CPU_AND_GPU)
    elif device == "cpu":
        wu_model = ct.models.MLModel('jacobi1024_model_fp16_100.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY)
    elif device == "all":
        wu_model = ct.models.MLModel('jacobi1024_model_fp16_100.mlpackage', compute_units=ct.ComputeUnit.ALL)

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
        
