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

import argparse
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

def main(matrix_size, datatype):
    model = MyMachine()
    model.eval()
    
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

    print("--------------------------")
    print("Creating matrix...")
    print("--------------------------\n")
    example_A = torch.rand(matrix_size, matrix_size, dtype=torchfloat)
    example_B = torch.rand(matrix_size, matrix_size, dtype=torchfloat) 

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
        inputs=[ct.TensorType(shape=example_A.shape, dtype=npfloat), ct.TensorType(shape=example_B.shape, dtype=npfloat)],
        outputs=[ct.TensorType(dtype=npfloat)],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ctfloat
    )

    # Export from program
    #exported_program = torch.export.export(model, (example_A, example_B))
    #model_from_export = ct.convert(exported_program)

    print("--------------------------")
    print("Saving the model...")
    print("--------------------------\n")
    model_from_trace.save(f"matmul{matrix_size}_model_{datatype}.mlpackage")


    print("--------------------------")
    print("Loading the model...")
    print("--------------------------\n")
    mlmodel = ct.models.MLModel(f"matmul{matrix_size}_model_{datatype}.mlpackage", compute_units=ct.ComputeUnit.ALL)

    print("--------------------------")
    print("Testing the model...")
    print("--------------------------\n")
    input_dict = {'A': example_A, 'B': example_B}
    result = mlmodel.predict(input_dict)

    print("+++ OK +++")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="python matmul.py matrix_size datatype.")
    
    # Add arguments
    parser.add_argument("matrix_input", type=int, help="matrix size in integer")
    parser.add_argument("datatype_input", type=str, help="datatype: fp32/fp16")
    
    # Parse the arguments
    args = parser.parse_args()

    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.matrix_input is None or args.datatype_input is None:
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Call the main function with provided arguments
        main(args.matrix_input, args.datatype_input)   
