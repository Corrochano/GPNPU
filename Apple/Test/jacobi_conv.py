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

import torch
from torch import nn
import coremltools as ct
import numpy as np
import torch.nn.functional as F

class JacobiMachine(nn.Module):
    def __init__(self, nt=100):
        super(JacobiMachine, self).__init__()
        self.nt = torch.tensor(nt)

    def forward(self, X, Y):    
      x = torch.exp(torch.mul(-50, torch.add(torch.pow((X - 0.5), 2), torch.pow((Y - 0.5), 2))))
      x = x.unsqueeze(0).unsqueeze(0) # Channel and batch size. Necessary for conv layer
      x_prev = x.clone()
    
      # Define the 3x3 kernel
      kernel = torch.tensor([[0.0, 0.25, 0.0],
                            [0.25, 0.0, 0.25],
                            [0.0, 0.25, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)

      mask = torch.ones_like(x)
      mask[:, :, 0, :] = 0        # Top boundary
      mask[:, :, -1, :] = 0       # Bottom boundary
      mask[:, :, :, 0] = 0        # Left boundary
      mask[:, :, :, -1] = 0       # Right boundary

      i = torch.tensor(0)
      
      while torch.ne(i, self.nt): # The for add can't go to the ane
          x_prev = x.clone()

          x_next = F.conv2d(x_prev, kernel, padding=1) 
                    
          x = x_next * mask

          # Check for convergence: max difference between u and u_prev
          diff = torch.max(torch.abs(x - x_prev))
          
          i = torch.add(i, 1)
      
      return x

jacobiModel = JacobiMachine()
jacobiModel.eval()

nx=1000
ny=1000
dt=0.001
alpha=0.01 

dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)  # spatial step sizes

x = torch.linspace(0, 1, nx, dtype=torch.float32)
y = torch.linspace(0, 1, ny, dtype=torch.float32)
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
    inputs=[ct.TensorType(shape=X.shape, dtype=np.float32), ct.TensorType(shape=Y.shape, dtype=np.float32)],
)


# Export from program
#exported_jacobi = torch.export.export(jacobiModel, (X, Y))
#jacobi_from_export = ct.convert(exported_jacobi, compute_units=ct.ComputeUnit.CPU_AND_NE)

print("--------------------------")
print("Saving the model...")
print("--------------------------\n")
jacobi_from_trace.save("conv_jacobi_WhileComplete.mlpackage")

print("--------------------------")
print("Loading the model...")
print("--------------------------\n")
mlmodel = ct.models.MLModel("conv_jacobi.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)

print("--------------------------")
print("Model input description:")
print("--------------------------\n")
print(mlmodel.get_spec().description.input)

print("--------------------------")
print("Testing the model...")
print("--------------------------\n")
input_dict = {'X': X, 'Y': Y} # If I write capital letters, there's an error 

result = mlmodel.predict(input_dict)

print("+++ OK +++")

