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

def main(size, runtimes, datatype, device):
   
   # Load the Core ML model
   if size.is_integer():
      ssize = str(int(size))
      matrix_size = int(size)

   else:
      ssize = str(size)
      matrix_size = int(size)

   if datatype == "fp64":
      torchfloat = torch.float64
   elif datatype == "fp32":
      torchfloat = torch.float32
   elif datatype == "fp16":
      torchfloat = torch.float16
   else:
      print('Error datatype not available!!!!')

   print("[INFO] Loading the model...")

   model_filename = f"matmul{(ssize)}_model_{datatype}.mlpackage"
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

   print("[INFO] Creating input matrix...")
   # Create random input matrix
   example_A = torch.rand(matrix_size, matrix_size, dtype=torchfloat)
   example_B = torch.rand(matrix_size, matrix_size, dtype=torchfloat) 
   
   # Prepare inputs for the model
   input_dict = {'A': example_A, 'B': example_B}

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

   flops = 2 * (matrix_size ** 3)
   gflops = flops / (10**9)  # Convert to GFLOPs
   gflops_per_second = gflops / elapsed_time 
   
   min_gflops_per_second = gflops / minTime 
   
   print("****************************************************************************************************************************")
   print(f"Matmul of size {matrix_size}x{matrix_size} in {datatype} took {elapsed_time:.4f} seconds.")
   print(f"Performance: {gflops_per_second:.2f} GFLOPs/s")
   print(f"Max Performance took {minTime:.4f} seconds with {min_gflops_per_second:.2f} GFLOPs/s")
   print("****************************************************************************************************************************")
   # Inspect the Core ML model to view input and output names
#   print("Model Inputs:", mlmodel.input_description)
#   print("Model Outputs:", mlmodel.output_description)

   # Use the actual output name printed from the model inspection
#   output_key = list(result.keys())[0]  # Access the first output if unsure
#   output = result[output_key]
#   print("A:", A)
#   print("B:", B)
#   print("Output:", output)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="python run_matmul.py matrix_size number_times_executed datatype device number_of_iterations.")

    # Add arguments
    parser.add_argument("float_input", type=float, help="matrix size in float")
    parser.add_argument("int_input", type=int, help="number of times executed")
    parser.add_argument("datatype_input", type=str, help="datatype: fp32/fp16")
    parser.add_argument("device_input", type=str, help="device: cpu/gpu/ane/all")

    # Parse the arguments
    args = parser.parse_args()

    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.float_input is None or args.int_input is None or args.datatype_input is None or args.device_input is None:
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Call the main function with provided arguments
        main(args.float_input, args.int_input, args.datatype_input, args.device_input)
