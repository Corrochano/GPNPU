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
import os
import argparse
import matplotlib.pyplot as plt

def read_file(fileName):
    f = open(fileName)

    text = f.read()

    lines = text.split('CPU 11 down residency:   ')

    f.close()
    
    # Save each device data
    cpu_power = []
    gpu_power = []
    ane_power = []
    total_power = []

    for i, line in enumerate(lines):
        if i != 0:
            cpu_power.append(int(line.split('\n')[1].split(' ')[2]))
            gpu_power.append(int(line.split('\n')[2].split(' ')[2]))
            ane_power.append(int(line.split('\n')[3].split(' ')[2]))
            total_power.append(int(line.split('\n')[4].split(' ')[7]))
            
    mean = sum(total_power) / len(total_power)
    cpu_mean = sum(cpu_power) / len(cpu_power)
    gpu_mean = sum(gpu_power) / len(gpu_power)
    ane_mean = sum(ane_power) / len(ane_power)
    
    maxim = max(total_power)
    cpu_max = max(cpu_power)
    gpu_max = max(gpu_power)
    ane_max = max(ane_power)    
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Mean total power on " + fileName + ':', mean )
    print("Mean CPU total power on " + fileName + ':', cpu_mean )
    print("Mean GPU total power on " + fileName + ':', gpu_mean )
    print("Mean ANE total power on " + fileName + ':', ane_mean )
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Max total power on " + fileName + ':', maxim )
    print("Max CPU total power on " + fileName + ':', cpu_max )
    print("Max GPU total power on " + fileName + ':', gpu_max )
    print("Max ANE total power on " + fileName + ':', ane_max )
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
    return cpu_power, gpu_power, ane_power

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="python3 graphic16vs32.py file16 mode")

    # Add arguments
    parser.add_argument("file_input_16", type=str, help="File with the fp16 data to process")
    parser.add_argument("mode_input", type=str, help="Execution mode (CPU/GPU/ANE/ALL)")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.file_input_16 is None or args.mode_input is None:
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Read the specified files
        cpu_16, gpu_16, ane_16 = read_file(args.file_input_16)
        
