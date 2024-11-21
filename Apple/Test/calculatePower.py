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

import matplotlib.pyplot as plt


def graphics(values):
    # Create x and y axis
    x = range(len(values))
    y = values
    
    # Adjust size
    plt.figure(figsize=(500, 500))
    
    # Generate graphic
    plt.plot(x, y)
    
    plt.yticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000])

    # Title and labels
    plt.title('Energy consumption')
    plt.xlabel('Time (s)')
    plt.ylabel('Consumption (mW)')

    #plt.legend()
    plt.savefig('cpu.png')
    #plt.show()
    plt.clf()

f = open('metrics.txt')

text = f.read()

lines = text.split('System instructions per clock:')

f.close()

cpu_power = []
gpu_power = []
ane_power = []

for i, line in enumerate(lines):
    if i != 0:
        cpu_power.append(line.split('\n')[1].split(' ')[2])
        gpu_power.append(line.split('\n')[2].split(' ')[2])
        ane_power.append(line.split('\n')[3].split(' ')[2])

graphics(cpu_power)


