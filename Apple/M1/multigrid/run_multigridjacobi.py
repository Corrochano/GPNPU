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
import numpy as np
import coremltools as ct
import time
import math
import os

def main(size, runtimes, datatype, device, iterations):
   
   # Load the Core ML model
   if size.is_integer():
      ssize = str(int(size))
      grid_size = int(size)

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

   model_filename = f"jacobi{(ssize)}_model_{datatype}_{iterations}.mlpackage"
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
   nx=grid_size
   ny=grid_size
   dt=0.001
   alpha=0.01 

   dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)  # spatial step sizes

   x = torch.linspace(0, 1, nx, dtype=npfloat)
   y = torch.linspace(0, 1, ny, dtype=npfloat)
   X, Y = torch.meshgrid(x, y)
   
   x = torch.exp(torch.mul( # Take the value of X
                        -50, 
                        torch.add(torch.pow((X - 0.5), 2), torch.pow((Y - 0.5), 2))
                    )).to(npfloat)
   x = x.unsqueeze(0).unsqueeze(0)    
    
   # Create masks
   mask = torch.ones_like(x, dtype=npfloat)
   mask[:, :, 0, :] = 0
   mask[:, :, -1, :] = 0
   mask[:, :, :, 0] = 0
   mask[:, :, :, -1] = 0    

   # Define num_levels
   num_levels = 9
    
   masks = [mask]
    
   for _ in range(num_levels):# precalculate masks
       masks.append(nn.AvgPool2d(kernel_size=2)(masks[-1]).to(npfloat))    
   
   # Prepare inputs for the model
   input_dict = {'X': X, 'Y': Y, 'Mask1': masks[0], 'Mask2': masks[1], 'Mask3': masks[2], 'Mask4': masks[3], 'Mask5': masks[4], 'Mask6': masks[5], 'Mask7': masks[6], 'Mask8': masks[7], 'Mask9': masks[8],
    'Mask10': masks[9]}
   

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
   
   # NEED TO ADJUST THE FORMULA
   '''
   # Calculate GFLOPs for jacobi
   init_flops = 7 * (grid_size ** 2) # Initial operation outside the while
   # While flops
   conv_flops = iterations * (17 * (grid_size ** 2)) # 3x3 convolution has 17 operations
   calculateNext_flops = grid_size ** 2 # Only a mul op
   diff_flops = 3 * (grid_size ** 2) # max, abs and sub (not sure about the max needs to be into account)
   iAdd_flops = 1 # i++ operation
   
   jacobiFlops = (init_flops + (iterations * (conv_flops + calculateNext_flops + diff_flops + iAdd_flops))) *  mlmodel.num_levels # total iterations on the while multiplied by all the ops on there
   '''
   '''
   jacobi_flops = iterations * (10 * (grid_size ** 2))
   
   restriction_flops = sum((grid_size // (2**l))**2 for l in range(9 - 1))
   
   interpolation_flops = sum(4 * (grid_size // (2**l))**2 for l in range(9 - 1))
   
   flops = jacobi_flops + restriction_flops + interpolation_flops

   gflops = flops / (10**9)  # Convert to GFLOPs
   gflops_per_second = gflops / elapsed_time 
   '''   

   print("****************************************************************************************************************************")
   print(f"Jacobi of size {grid_size}x{grid_size} with {iterations} iterations in {datatype} took {elapsed_time:.4f} seconds.")
   print(f"Max Performance took {minTime:.4f} seconds")
   print("****************************************************************************************************************************")
   
   if os.path.exists(f"jacobi_{grid_size}x{grid_size}_{device}.csv"):

        log_content = (
            f"{datatype};{elapsed_time:.4f};{minTime:.4f};"
        )

   else:
        log_content = (
            f"datatype;MeanTime;PeakTime;datatype;MeanTime;PeakTime;"
            f"Datatype;TotalMeanEnergy;cpuMeanEnergy;gpuMeanEnergy;aneMeanEnergy;"
            f"Datatype;TotalMaxEnergy;cpuMaxEnergy;gpuMaxEnergy;aneMaxEnergy;"
            f"Datatype;TotalMeanEnergy;cpuMeanEnergy;gpuMeanEnergy;aneMeanEnergy;"
            f"Datatype;TotalMaxEnergy;cpuMaxEnergy;gpuMaxEnergy;aneMaxEnergy;\n"
            f"{datatype};{elapsed_time:.4f};{minTime:.4f};"
        )


   log_filename = f"jacobi_{grid_size}x{grid_size}_{device}.csv"
   with open(log_filename, "a") as f:
       f.write(log_content)
   
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
