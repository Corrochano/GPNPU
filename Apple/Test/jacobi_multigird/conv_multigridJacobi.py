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

# To install: curl -L -O https://www.openssl.org/source/openssl-1.1.1u.tar.gz    https://github.com/hollance/neural-engine/issues/35   https://developer.apple.com/videos/play/wwdc2022/10027/ https://github.com/hollance/neural-engine/blob/master/docs/model-surgery.md

# To use asitop: sudo /Users/acorrochano/bin/pythonOld/bin/python3 -m asitop.asitop

import argparse
import torch
from torch import nn
import coremltools as ct
import numpy as np
import torch.nn.functional as F

class JacobiMachine(nn.Module):
    def __init__(self, nt=1000, datatype=torch.float32):
        super(JacobiMachine, self).__init__()
        self.datatype=datatype
        self.nt = torch.tensor(nt, dtype=self.datatype)

    def forward(self, X, Y):    
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
    
      
      # Define the 3x3 kernel
      kernel = torch.tensor([[0.0, 0.25, 0.0],
                            [0.25, 0.0, 0.25],
                            [0.0, 0.25, 0.0]], dtype=self.datatype).view(1, 1, 3, 3)
                            
      i = torch.tensor(0, dtype=self.datatype)
      
      while torch.ne(i, self.nt):
          x_prev = x.clone()

          x_next = F.conv2d(x_prev, kernel, padding=1)
                    
          x = x_next * mask

          # Check for convergence: max difference between u and u_prev
          diff = torch.max(torch.abs(x - x_prev))
          
          i = torch.add(i, 1)
      
      return x

def main(grid_size, iterations, dataType):

    nt = iterations
    datatype = dataType

    if datatype == "fp32":
       torchfloat = torch.float32
       npfloat = np.float32
       ctfloat = ct.precision.FLOAT32
    elif datatype == "fp16":
       torchfloat = torch.float16
       npfloat = np.float16
       ctfloat = ct.precision.FLOAT16
    else:
       print('Error datatype not available!!!!')

    ''' # fp64 not in core tools, only fp32, 16 and integer
    if datatype == "fp64":
       torchfloat = torch.float64
       npfloat = np.float64
       ctfloat = ct.precision.FLOAT64
    '''

    jacobiModel = JacobiMachine(nt,torchfloat)
    jacobiModel.eval()
    jacobiModel.float()

    nx=grid_size
    ny=grid_size
    dt=0.001
    alpha=0.01 

    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)  # spatial step sizes

    x = torch.linspace(0, 1, nx, dtype=torchfloat)
    y = torch.linspace(0, 1, ny, dtype=torchfloat)
    X, Y = torch.meshgrid(x, y)

    '''
    X = X.float()
    Y = Y.float()
    '''

    print("--------------------------")
    print("Testing the model:")
    output = jacobiModel(X, Y)
    print("--------------------------\n")

    print("--------------------------")
    print("Exporting the model...")
    print("--------------------------\n")

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # Export from trace
    traced_model = torch.jit.trace(jacobiModel, (X, Y))
    jacobi_from_trace = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=X.shape, dtype=npfloat), ct.TensorType(shape=Y.shape, dtype=npfloat)],
        outputs=[ct.TensorType(dtype=npfloat)],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ctfloat
    )

    # Export from program
    #exported_jacobi = torch.export.export(jacobiModel, (X, Y))
    #jacobi_from_export = ct.convert(exported_jacobi, compute_units=ct.ComputeUnit.CPU_AND_NE)

    print("--------------------------")
    print("Saving the model...")
    print("--------------------------\n")
    jacobi_from_trace.save(f"jacobi{nx//1000}k_model_{datatype}_{nt}.mlpackage")

    print("--------------------------")
    print("Loading the model...")
    print("--------------------------\n")
    mlmodel = ct.models.MLModel(f"jacobi{nx//1000}k_model_{datatype}_{nt}.mlpackage", compute_units=ct.ComputeUnit.ALL)

    print("--------------------------")
    print("Model input description:")
    print("--------------------------\n")
    print(mlmodel.get_spec().description.input)

    print("--------------------------")
    print("Testing the model...")
    print("--------------------------\n")
    input_dict = {'X': X, 'Y': Y}
    result = mlmodel.predict(input_dict)

    print("+++ OK +++")
    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="python jacobi_conv.py grid_size number_of_iterations datatype.")
    
    # Add arguments
    parser.add_argument("grid_input", type=int, help="grids size in integer")
    parser.add_argument("iteration_input", type=int, help="Number of iterations on the jacobi")
    parser.add_argument("datatype_input", type=str, help="datatype: fp32/fp16")
    
    # Parse the arguments
    args = parser.parse_args()

    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.grid_input is None or args.iteration_input is None or args.datatype_input is None:
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Call the main function with provided arguments
        main(args.grid_input, args.iteration_input, args.datatype_input)    

