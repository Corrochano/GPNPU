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

def main(size, runtimes, datatype, device, iterations):
   
   # Load the Core ML model
   if size.is_integer():
      ssize = str(int(size))
      grid_size = int(size*1000)

   else:
      ssize = str(size)
      grid_size = int(size)

   if datatype == "fp64":
      npfloat = torch.float64
   elif datatype == "fp32":
      npfloat = torch.float32
   elif datatype == "fp16":
      npfloat = torch.float16
   else:
      print('Error datatype not available!!!!')

   print("[INFO] Loading the model...")

   model_filename = f"jacobi{(ssize)}k_model_{datatype}_{iterations}.mlpackage"
   if device == "ane":
      mlmodel = ct.models.MLModel(model_filename, compute_units=ct.ComputeUnit.CPU_AND_NE)
   elif device == "gpu":
      mlmodel = ct.models.MLModel(model_filename, compute_units=ct.ComputeUnit.CPU_AND_GPU)
   elif device == "cpu":
      mlmodel = ct.models.MLModel(model_filename, compute_units=ct.ComputeUnit.CPU_ONLY)
   elif device == "all":
      mlmodel = ct.models.MLModel(model_filename, compute_units=ct.ComputeUnit.ALL)
   else:
      print('Error device not found: cpu/gpu/ane')

   print("[INFO] Creating input grid...")
   # Create random input grid
   x = torch.linspace(0, 1, grid_size, dtype=npfloat)
   y = torch.linspace(0, 1, grid_size, dtype=npfloat)
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
   input_dict = {'X': x, 'X_prev': x_prev, 'Mask': mask}

   print("[INFO] Running inference...")

   # Run inference using the Core ML model
   start_time = time.time()
   for i in range(runtimes):
      result = mlmodel.predict(input_dict)
   end_time = time.time()
   elapsed_time = ( end_time - start_time ) / runtimes

   # Calculate GFLOPs
   # While flops
   flops = iterations * (10 * (grid_size ** 2)) # 3x3 convolution has 10 operations
   
   gflops = flops / (10**9)  # Convert to GFLOPs
   gflops_per_second = gflops / elapsed_time 
   print("****************************************************************************************************************************")
   print(f"Jacobi of size {grid_size}x{grid_size} with {iterations} iterations in {datatype} took {elapsed_time:.4f} seconds.")
   print(f"Performance: {gflops_per_second:.2f} GFLOPs/s")
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
    parser = argparse.ArgumentParser(description="python script.py grid_size number_times_executed datatype device number_of_iterations.")

    # Add arguments
    parser.add_argument("float_input", type=float, help="grids size in float")
    parser.add_argument("iteration_input", type=int, help="Number of iterations on the jacobi")
    parser.add_argument("int_input", type=int, help="number of times executed")
    parser.add_argument("datatype_input", type=str, help="datatype: fp32/fp16")
    parser.add_argument("device_input", type=str, help="device: cpu/gpu/ane/all")

    # Parse the arguments
    args = parser.parse_args()

    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.float_input is None or args.int_input is None or args.datatype_input is None or args.device_input is None or args.iteration_input is None:
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Call the main function with provided arguments
        main(args.float_input, args.int_input, args.datatype_input, args.device_input, args.iteration_input)
