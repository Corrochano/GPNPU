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
from sklearn.metrics import r2_score

class MyMachine(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, A, B):
        x = torch.matmul(A, B)
        return x

model = MyMachine()

print("--------------------------")
print("Creating matrix...")
print("--------------------------\n")
# Trace the model with random data.
example_A = torch.rand(1, 256)
example_B = torch.rand(256, 1) 
#traced_model = torch.jit.trace(model, (example_A, example_B))

print("--------------------------")
print("Testing the model:")
output = model(example_A, example_B)
'''
print()
print("Matrix A:", example_A)
print("Matrix B:", example_B)
print()
print("Result:", output)
'''
print("--------------------------\n")

print("--------------------------")
print("Exporting the model...")
print("--------------------------\n")
# Export the model
exported_program = torch.export.export(model, (example_A, example_B))

model_from_export = ct.convert(exported_program, compute_units=ct.ComputeUnit.CPU_AND_NE)

print("--------------------------")
print("Saving the model...")
print("--------------------------\n")
model_from_export.save("newmodel_from_export.mlpackage")

print("--------------------------")
print("Loading the model...")
print("--------------------------\n")
mlmodel = ct.models.MLModel("newmodel_from_export.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)

print("--------------------------")
print("Testing the model...")
print("--------------------------\n")
input_dict = {'a': example_A, 'b': example_B} # If I write capital letters, there's an error 
result = mlmodel.predict(input_dict)

print("+++ OK +++")

