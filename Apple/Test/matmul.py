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

# https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html
import torch
from torch import nn
import coremltools as ct
import numpy as np

class MyMachine(nn.Module):
    def __init__(self):
        super(MyMachine, self).__init__()
        
    def forward(self, A, B):
        x = torch.matmul(A, B)
        return x

model = MyMachine()
model.eval()

print("--------------------------")
print("Creating matrix...")
print("--------------------------\n")
example_A = torch.rand(1000, 512, dtype=torch.float64)
example_B = torch.rand(512, 1000, dtype=torch.float64) 
'''
print("--------------------------")
print("Testing the model:")
output = model(example_A, example_B)
print("--------------------------\n")

print("--------------------------")
print("Exporting the model...")
print("--------------------------\n")

# Export from trace
traced_model = torch.jit.trace(model, (example_A, example_B))
model_from_trace = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_A.shape, dtype=np.float64), ct.TensorType(shape=example_B.shape, dtype=np.float64)],
)

# Export from program
#exported_program = torch.export.export(model, (example_A, example_B))
#model_from_export = ct.convert(exported_program)

print("--------------------------")
print("Saving the model...")
print("--------------------------\n")
model_from_trace.save("newmodel_from_export.mlpackage")
'''

print("--------------------------")
print("Loading the model...")
print("--------------------------\n")
mlmodel = ct.models.MLModel("newmodel_from_export.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)

print("--------------------------")
print("Testing the model...")
print("--------------------------\n")
while True:
    input_dict = {'A': example_A, 'B': example_B} # If I write capital letters, there's an error

    result = mlmodel.predict(input_dict)

print("+++ OK +++")

