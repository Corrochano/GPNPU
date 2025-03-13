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
        self.nt = nt 

    def forward(self, X, Y):    
      x = torch.exp(torch.mul(-50, torch.add(torch.pow((X - 0.5), 2), torch.pow((Y - 0.5), 2)))) # Run into ANE
      
      
      x_prev = x.clone()

      for t in range(self.nt):
          x_prev = x.clone()
          
          x_next = x.clone()
          x_next[1:-1, 1:-1] = torch.mul(
              0.25,
              torch.add(
                  torch.add(x_prev[1:-1, 2:], x_prev[1:-1, :-2]),
                  torch.add(x_prev[2:, 1:-1], x_prev[:-2, 1:-1])
              )
          )
          
          mask = torch.ones_like(x_next)
          mask[:, 0] = 0
          mask[:, -1] = 0
          mask[0, :] = 0
          mask[-1, :] = 0
          
          x_next *= mask
          
          x = x_next.clone()
          
          # Check for convergence: max difference between u and u_prev
          diff = torch.max(torch.abs(x - x_prev))
          
      return x

jacobiModel = JacobiMachine()
jacobiModel.eval()

nx=200
ny=200
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
jacobi_from_trace.save("jacobi.mlpackage")

print("--------------------------")
print("Loading the model...")
print("--------------------------\n")
mlmodel = ct.models.MLModel("jacobi.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)

print("--------------------------")
print("Testing the model...")
print("--------------------------\n")
while True:
    input_dict = {'X': X, 'Y': Y} # If I write capital letters, there's an error

#while True:
result = mlmodel.predict(input_dict)

print("+++ OK +++")

