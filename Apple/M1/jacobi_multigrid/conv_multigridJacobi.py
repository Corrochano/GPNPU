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
    def __init__(self, nt=1000, num_levels=9, datatype=torch.float32):
        super(JacobiMachine, self).__init__()
        self.datatype=datatype
        self.num_levels = num_levels
        self.nt = torch.tensor(nt, dtype=self.datatype)

    def restriction(self, residual): # Applied when we want to convert from finegrid to coarsergrid
        return nn.AvgPool2d(kernel_size=2)(residual)
    
    def interpolate(self, f, target_size): # Applied when we want to convert from coarsergrid to finegrid
        return F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)

    def jacobi(self, Z, mask): # Jacobi method   
      x = Z
      x_prev = x.clone()
      
      '''
      mask = torch.ones_like(x)
      
      mask[:, :, 0, :] = 0
      mask[:, :, -1, :] = 0
      mask[:, :, :, 0] = 0
      mask[:, :, :, -1] = 0
      '''
              
      # Define the 3x3 kernel
      kernel = torch.tensor([[0.0, 0.25, 0.0],
                            [0.25, 0.0, 0.25],
                            [0.0, 0.25, 0.0]], dtype=self.datatype).view(1, 1, 3, 3)
                            
      i = torch.tensor(0, dtype=self.datatype)
      
      print(mask.shape  )
      
      while torch.ne(i, self.nt):
          x_prev = x.clone()
          x_next = F.conv2d(x_prev, kernel, padding=1)            
          x = x_next * mask
          
          i = torch.add(i, 1)
      
      return x

    def forward(self, X, Y, Mask1, Mask2, Mask3, Mask4, Mask5, Mask6, Mask7, Mask8, Mask9, Mask10):
        Masks = [Mask1,Mask2,Mask3,Mask4,Mask5,Mask6,Mask7,Mask8,Mask9,Mask10]
        x = torch.exp(torch.mul( # Take the value of X
                        -50, 
                        torch.add(torch.pow((X - 0.5), 2), torch.pow((Y - 0.5), 2))
                    )).to(self.datatype)
        x = x.unsqueeze(0).unsqueeze(0) # Channel and batch size. Necessary for conv layer

        '''
        mask = torch.ones_like(x, dtype=self.datatype)
        mask[:, :, 0, :] = 0
        mask[:, :, -1, :] = 0
        mask[:, :, :, 0] = 0
        mask[:, :, :, -1] = 0
        '''
        
        # Define the 3x3 kernel
        kernel = torch.tensor([[0.0, 0.25, 0.0],
                               [0.25, 0.0, 0.25],
                               [0.0, 0.25, 0.0]], dtype=self.datatype).view(1, 1, 3, 3)
        
        # Define num_levels
        num_levels = self.num_levels
        
        '''
        masks = [mask]
        
        for _ in range(num_levels):# precalculate masks
            masks.append(self.restriction(masks[-1]).to(self.datatype))
        
        '''
        grids = [x]
        
        current_mask = Masks[0]
        
        # Downward phase
        for level in range(1, num_levels):
            masked_input = torch.mul(grids[level - 1] , (Masks[level - 1] > 0))
            unmasked_input = torch.mul(grids[level - 1] , (1 - Masks[level - 1]))

            masked_output = F.conv2d(masked_input, kernel, padding=1)
            unmasked_output = F.conv2d(unmasked_input, kernel, padding=1)

            residual = masked_output + unmasked_output
            
            coarse_residual = self.restriction(residual)  # Restrict to coarser grid
        
            grids.append(coarse_residual)  # Store
        
        # Solve on the coarsest grid
        coarse_solution = grids[-1]
        coarse_solution = self.jacobi(coarse_solution, Mask9)  # Solve the classic jacobi there
        grids[-1] = coarse_solution  # Update store solution     
               
        # Upward phase
        for level in range(num_levels - 2, -1, -1):

            target_size = grids[level].shape[-2:]
          
            fine_solution = self.interpolate(grids[level + 1], target_size=target_size)  # Interpolate to finer gridclear
                        
            fine_solution = torch.add( fine_solution, grids[level] ) # += grids[level]  # Add correction to finer grid    
            
            masked_input = torch.mul(fine_solution , (Masks[level] > 0))
            unmasked_input = torch.mul(fine_solution , (1 - Masks[level]))

            masked_output = F.conv2d(masked_input, kernel, padding=1)
            unmasked_output = F.conv2d(unmasked_input, kernel, padding=1)

            fine_solution = masked_output + unmasked_output
            
        # Final refinement on the finest grid        
        while len(grids) > 1:
            grids.pop()
        
        final_solution = grids.pop()
        final_solution = self.jacobi(final_solution, Mask1)  # Final Jacobi iterations
    
        return final_solution
        
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

    jacobiModel = JacobiMachine(nt=nt, datatype=torchfloat)
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
    
    
    
    
    
    
    x = torch.exp(torch.mul( # Take the value of X
                        -50, 
                        torch.add(torch.pow((X - 0.5), 2), torch.pow((Y - 0.5), 2))
                    )).to(torchfloat)
    x = x.unsqueeze(0).unsqueeze(0)    
    
    # Create masks
    mask = torch.ones_like(x, dtype=torchfloat)
    mask[:, :, 0, :] = 0
    mask[:, :, -1, :] = 0
    mask[:, :, :, 0] = 0
    mask[:, :, :, -1] = 0    

    # Define num_levels
    num_levels = 9
    
    masks = [mask]
    
    for _ in range(num_levels):# precalculate masks
        masks.append(nn.AvgPool2d(kernel_size=2)(masks[-1]).to(torchfloat))    
    
    
    print("Number of masks:", len(masks))
    
    

    '''
    X = X.float()
    Y = Y.float()
    '''

    print("--------------------------")
    print("Testing the model:")
    output = jacobiModel(X, Y, masks[0], masks[1], masks[2], masks[3], masks[4], masks[5], masks[6], masks[7], masks[8], masks[9])
    print("--------------------------\n")

    print("--------------------------")
    print("Exporting the model...")
    print("--------------------------\n")

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # Export from trace
    traced_model = torch.jit.trace(jacobiModel, (X, Y, masks[0], masks[1], masks[2], masks[3], masks[4], masks[5], masks[6], masks[7], masks[8], masks[9]))
    jacobi_from_trace = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=X.shape, dtype=npfloat), ct.TensorType(shape=Y.shape, dtype=npfloat), ct.TensorType(shape=masks[0].shape, dtype=npfloat), ct.TensorType(shape=masks[1].shape, dtype=npfloat), 
        ct.TensorType(shape=masks[2].shape, dtype=npfloat), ct.TensorType(shape=masks[3].shape, dtype=npfloat), ct.TensorType(shape=masks[4].shape, dtype=npfloat), 
        ct.TensorType(shape=masks[5].shape, dtype=npfloat), ct.TensorType(shape=masks[6].shape, dtype=npfloat), ct.TensorType(shape=masks[7].shape, dtype=npfloat), ct.TensorType(shape=masks[8].shape, dtype=npfloat), ct.TensorType(shape=masks[9].shape, dtype=npfloat)],
        outputs=[ct.TensorType(dtype=npfloat)],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ctfloat
    )
    # inputs=[ct.TensorType(shape=X.shape, dtype=npfloat), ct.TensorType(shape=Y.shape, dtype=npfloat), len(masks), ), dtype=npfloat)],

    # Export from program
    #exported_jacobi = torch.export.export(jacobiModel, (X, Y))
    #jacobi_from_export = ct.convert(exported_jacobi, compute_units=ct.ComputeUnit.CPU_AND_NE)

    print("--------------------------")
    print("Saving the model...")
    print("--------------------------\n")
    jacobi_from_trace.save(f"jacobi{nx}_model_{datatype}_{nt}.mlpackage")

    print("--------------------------")
    print("Loading the model...")
    print("--------------------------\n")
    mlmodel = ct.models.MLModel(f"jacobi{nx}_model_{datatype}_{nt}.mlpackage", compute_units=ct.ComputeUnit.ALL)

    print("--------------------------")
    print("Model input description:")
    print("--------------------------\n")
    print(mlmodel.get_spec().description.input)

    print("--------------------------")
    print("Testing the model...")
    print("--------------------------\n")
    input_dict = {'X': X, 'Y': Y, 'Mask1': masks[0], 'Mask2': masks[1], 'Mask3': masks[2], 'Mask4': masks[3], 'Mask5': masks[4], 'Mask6': masks[5], 'Mask7': masks[6], 'Mask8': masks[7], 'Mask9': masks[8],
    'Mask10': masks[9]}
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

