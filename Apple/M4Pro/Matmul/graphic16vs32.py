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

def graphics(cpu_16, gpu_16, ane_16, cpu_32, gpu_32, ane_32, size, mode, folder_path):
    x = list(range(len(cpu_16)))
    x = list(map(lambda i: i * 100, x))
    y = list(range(len(gpu_16)))
    y = list(map(lambda i: i * 100, y))
    z = list(range(len(ane_16)))
    z = list(map(lambda i: i * 100, z))

    x_32 = list(range(len(cpu_32)))
    x_32 = list(map(lambda i: i * 100, x_32))
    y_32 = list(range(len(gpu_32)))
    y_32 = list(map(lambda i: i * 100, y_32))
    z_32 = list(range(len(ane_32)))
    z_32 = list(map(lambda i: i * 100, z_32))  
        
    plt.figure(figsize=(45,33))

    # Generate graphic
    plt.plot(x, cpu_16, label='FP16 CPU Consumption', color='#0000FF')
    plt.plot(y, gpu_16, label='FP16 GPU Consumption', color='#FF5733')
    plt.plot(z, ane_16, label='FP16 ANE Consumption', color='#FFC300')

    plt.plot(x_32, cpu_32, label='FP32 CPU Consumption', color='#0000FF', linestyle='--')
    plt.plot(y_32, gpu_32, label='FP32 GPU Consumption', color='#FF5733', linestyle='--')
    plt.plot(z_32, ane_32, label='FP32 ANE Consumption', color='#FFC300', linestyle='--')    
    
    
    plt.title(f'{size} FP16 vs FP32 Energy consumption executed with {mode}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Consumption (mW)')
    
    plt.legend()
    plt.savefig(os.path.join(folder_path, f'16vs32{size}_mode{mode}.svg'))
    plt.clf()

def read_file(fileName):
    f = open(fileName)

    text = f.read()

    lines = text.split('CPU 11 down residency:')

    f.close()
    
    # Save each device data
    cpu_power = []
    gpu_power = []
    ane_power = []
    total_power = []

    for i, line in enumerate(lines):
        if i != 0:
            cpu_power.append(int(line.split('\n')[2].split(' ')[2]))
            gpu_power.append(int(line.split('\n')[3].split(' ')[2]))
            ane_power.append(int(line.split('\n')[4].split(' ')[2]))
            total_power.append(int(line.split('\n')[5].split(' ')[7]))
            
    mean = sum(total_power) / len(total_power)
    cpu_mean = sum(cpu_power) / len(cpu_power)
    gpu_mean = sum(gpu_power) / len(gpu_power)
    ane_mean = sum(ane_power) / len(ane_power)
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Mean total power on " + fileName + ':', mean )
    print("Mean CPU total power on " + fileName + ':', cpu_mean )
    print("Mean GPU total power on " + fileName + ':', gpu_mean )
    print("Mean ANE total power on " + fileName + ':', ane_mean )
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
    return cpu_power, gpu_power, ane_power

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="python3 graphic16vs32.py file16 file32 size mode")

    # Add arguments
    parser.add_argument("file_input_16", type=str, help="File with the fp16 data to process")
    parser.add_argument("file_input_32", type=str, help="File with the fp32 data to process")
    parser.add_argument("size_input", type=str, help="Size of the matrix")
    parser.add_argument("mode_input", type=str, help="Execution mode (CPU/GPU/ANE/ALL)")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.file_input_16 is None or args.size_input is None or args.file_input_32 is None or args.mode_input is None:
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Read the specified files
        cpu_16, gpu_16, ane_16 = read_file(args.file_input_16)
        cpu_32, gpu_32, ane_32 = read_file(args.file_input_32)        
       
        # Save all the graphics
        first_path = os.path.join(os.getcwd(), f"matmul")
        second_path = os.path.join(first_path, f"{args.size_input}")

        if not os.path.isdir(first_path):
                os.mkdir(first_path)
        if not os.path.isdir(second_path):
                os.mkdir(second_path)
        
        graphics(cpu_16, gpu_16, ane_16, cpu_32, gpu_32, ane_32, args.size_input, args.mode_input, second_path)
        
