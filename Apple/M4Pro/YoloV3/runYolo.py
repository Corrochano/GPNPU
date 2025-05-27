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
import numpy as np
import coremltools as ct
import time
import math
from PIL import Image

def main(runtimes, device):
   # Load the Core ML model
   print("[INFO] Loading the model...")

   torchfloat = torch.float16

   model_filename = f"YOLOv3Tiny.mlmodel"
   if device == "ane":
      mlmodel = ct.models.MLModel(model_filename, compute_units=ct.ComputeUnit.CPU_AND_NE)
   elif device == "gpu":
      mlmodel = ct.models.MLModel(model_filename, compute_units=ct.ComputeUnit.CPU_AND_GPU)
   elif device == "cpu":
      mlmodel = ct.models.MLModel(model_filename, compute_units=ct.ComputeUnit.CPU_ONLY)
   elif device == "all":
      mlmodel = ct.models.MLModel(model_filename, compute_units=ct.ComputeUnit.ALL)
   else:
      print('Error device not found: cpu/gpu/ane/all')

   print("[INFO] Creating input...")
   # Create random input matrix
   width, height = 416, 416

   img = np.random.rand(height, width, 3).astype(np.uint8)
   
   example = Image.fromarray(img)
   
   input_dict = {'image': example}
   
   print("[INFO] Running inference...")
      
   minTime = math.inf
   
   # Run inference using the Core ML model
   start_time = time.time()
   for i in range(runtimes):
      start_local_time = time.time()
      result = mlmodel.predict(input_dict)
      end_local_time = time.time()
      local_elapsed_time = ( end_local_time - start_local_time )
      minTime = minTime if (minTime < local_elapsed_time) else local_elapsed_time
   end_time = time.time()
   elapsed_time = ( end_time - start_time ) / runtimes
   
   print("****************************************************************************************************************************")
   print(f"Yolov3 took {elapsed_time:.4f} seconds.")
   print(f"Max Performance took {minTime:.4f} seconds.")
   print("****************************************************************************************************************************")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="python runYolo.py number_times_executed datatype device number_of_iterations.")

    # Add arguments
    parser.add_argument("int_input", type=int, help="number of times executed")
    parser.add_argument("device_input", type=str, help="device: cpu/gpu/ane/all")

    # Parse the arguments
    args = parser.parse_args()

    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.int_input is None or args.device_input is None:
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Call the main function with provided arguments
        main(args.int_input, args.device_input)

