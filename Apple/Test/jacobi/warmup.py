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

    print("[INFO] Creating input matrix...")
    warmup_A = torch.rand(1000, 512, dtype=torch.float16)
    warmup_B = torch.rand(512, 1000, dtype=torch.float16) 
    warmup_dict = {'A': warmup_A, 'B': warmup_B}

    print("[INFO] Loading warmup model...")
    if device == "ane":
        wu_model = ct.models.MLModel('matmulfp16.mlpackage', compute_units=ct.ComputeUnit.CPU_AND_NE)
    elif device == "gpu":
        wu_model = ct.models.MLModel('matmulfp16.mlpackage', compute_units=ct.ComputeUnit.CPU_AND_GPU)
    elif device == "cpu":
        wu_model = ct.models.MLModel('matmulfp16.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY)
    elif device == "all":
        wu_model = ct.models.MLModel('matmulfp16.mlpackage', compute_units=ct.ComputeUnit.ALL)

    print("[INFO] Running warmup...")
    for i in range(1000):
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
        
