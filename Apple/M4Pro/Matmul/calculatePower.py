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


def graphics(values, device, size, precision, mode, folder_path):
    # Create x and y axis
    x = list(range(len(values)))
    x = list(map(lambda i: i * 100, x))
    y = values
    
    plt.figure(figsize=(45,33))
    
    # Generate graphic
    plt.plot(x, y)

    # Title and labels
    plt.title(f'{size} {device} Energy consumption with {precision} executed with {mode}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Consumption (mW)')

    #plt.legend()
    plt.savefig(os.path.join(folder_path, f'{size}_{device}_{precision}_mode{mode}.svg'))
    plt.clf()

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="python3 calculatePower.py file size")

    # Add arguments
    parser.add_argument("file_input", type=str, help="File with the data to process")
    parser.add_argument("size_input", type=str, help="Size of the matrix")
    parser.add_argument("precision_input", type=str, help="Precision of the model (fp32/fp16)")
    parser.add_argument("mode_input", type=str, help="Execution mode (CPU/GPU/ANE/ALL)")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.file_input is None or args.size_input is None or args.precision_input is None or args.mode_input is None:
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Read the specified file
        f = open(args.file_input)

        text = f.read()

        lines = text.split('System instructions per clock:')

        f.close()
        
        # Save each device data
        cpu_power = []
        gpu_power = []
        ane_power = []

        for i, line in enumerate(lines):
            if i != 0:
                cpu_power.append(int(line.split('\n')[1].split(' ')[2]))
                gpu_power.append(int(line.split('\n')[2].split(' ')[2]))
                ane_power.append(int(line.split('\n')[3].split(' ')[2]))
                
       
        # Save all the graphics
        first_path = os.path.join(os.getcwd(), f"matmul")
        second_path = os.path.join(first_path, f"{args.size_input}")
        third_path = os.path.join(second_path, f"{args.precision_input}")
        folder_path = os.path.join(third_path, f"{args.mode_input}")

        if not os.path.isdir(first_path):
                os.mkdir(first_path)
        if not os.path.isdir(second_path):
                os.mkdir(second_path)
        if not os.path.isdir(third_path):
                os.mkdir(third_path)
        if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
        
        graphics(cpu_power, "CPU", args.size_input, args.precision_input, args.mode_input, folder_path)
        graphics(gpu_power, "GPU", args.size_input, args.precision_input, args.mode_input, folder_path)
        graphics(ane_power, "ANE", args.size_input, args.precision_input, args.mode_input, folder_path)
        
