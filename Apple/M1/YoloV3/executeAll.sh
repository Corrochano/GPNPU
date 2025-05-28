#!/bin/bash

#Copyright 2024 Álvaro Corrochano López
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

# Use example: sudo ./startJacobiTest.sh metrics.txt 1 100 1 fp32 ane
if [ $# -lt 1 ]; then
    echo "Usage: $0 <param1>"
    exit 1
fi

number_exec=$1

echo " *** Running CPU FP16***"

echo "powermetrics"
sudo rm -f cpu_16.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > cpu_16.txt &

echo " Running the matmul model..."
# Test to launch (exec)
/Users/acorrochano/bin/pyenv/bin/python3 runYolo.py $number_exec cpu

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/acorrochano/bin/pyenv/bin/python3 calculatePower.py cpu_16.txt cpu

echo " *** Running GPU FP16***"

echo "powermetrics"
sudo rm -f cpu_16.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > gpu_16.txt &

echo " Running the matmul model..."
# Test to launch (exec)
/Users/acorrochano/bin/pyenv/bin/python3 runYolo.py $number_exec gpu

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/acorrochano/bin/pyenv/bin/python3 calculatePower.py gpu_16.txt gpu

echo " *** Running ANE FP16***"

echo "powermetrics"
sudo rm -f cpu_16.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > ane_16.txt &

echo " Running the matmul model..."
# Test to launch (exec)
/Users/acorrochano/bin/pyenv/bin/python3 runYolo.py $number_exec ane

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/acorrochano/bin/pyenv/bin/python3 calculatePower.py ane_16.txt ane

