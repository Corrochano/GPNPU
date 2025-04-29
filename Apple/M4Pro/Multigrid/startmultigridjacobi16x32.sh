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

if [ $# -lt 2 ]; then
    echo "Usage: $0 <param1> <param2>"
    exit 1
fi

size=$1
number_exec=$2

echo " *** Running CPU FP16***"

echo " Running the warmup..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 warmup.py cpu

echo "powermetrics"
sudo rm -f cpu_16.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > cpu_16.txt &

echo " Running the jacobi model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 run_multigridjacobi.py $size 100 $number_exec fp16 cpu

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py cpu_16.txt ${size} fp16 cpu
/Users/alvarocorrochano/bin/.pyenv/bin/python3 allGraphic.py cpu_16.txt ${size} fp16 cpu
echo " Done!"

echo " *** Running CPU FP32***"

echo " Running the warmup..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 warmup.py cpu

echo "powermetrics"
sudo rm -f cpu_32.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > cpu_32.txt &

echo " Running the jacobi model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 run_multigridjacobi.py $size 100 $number_exec fp32 cpu

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py cpu_32.txt ${size} fp32 cpu
/Users/alvarocorrochano/bin/.pyenv/bin/python3 allGraphic.py cpu_32.txt ${size} fp32 cpu
echo " Done!"

echo " Saving vs metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 graphic16vs32.py cpu_16.txt cpu_32.txt ${size} cpu





echo " *** Running GPU FP16***"

echo " Running the warmup..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 warmup.py gpu

echo "powermetrics"
sudo rm -f gpu_16.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > gpu_16.txt &

echo " Running the jacobi model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 run_multigridjacobi.py $size 100 $number_exec fp16 gpu

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py gpu_16.txt ${size} fp16 gpu
/Users/alvarocorrochano/bin/.pyenv/bin/python3 allGraphic.py gpu_16.txt ${size} fp16 gpu
echo " Done!"

echo " *** Running GPU FP32***"

echo " Running the warmup..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 warmup.py gpu

echo "powermetrics"
sudo rm -f gpu_32.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > gpu_32.txt &

echo " Running the jacobi model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 run_multigridjacobi.py $size 100 $number_exec fp32 gpu

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py gpu_32.txt ${size} fp32 gpu
/Users/alvarocorrochano/bin/.pyenv/bin/python3 allGraphic.py gpu_32.txt ${size} fp32 gpu
echo " Done!"

echo " Saving vs metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 graphic16vs32.py gpu_16.txt gpu_32.txt ${size} gpu




echo " *** Running ANE FP16***"

echo " Running the warmup..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 warmup.py ane

echo "powermetrics"
sudo rm -f ane_16.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > ane_16.txt &

echo " Running the jacobi model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 run_multigridjacobi.py $size 100 $number_exec fp16 ane

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py ane_16.txt ${size} fp16 ane
/Users/alvarocorrochano/bin/.pyenv/bin/python3 allGraphic.py ane_16.txt ${size} fp16 ane
echo " Done!"

echo " *** Running ANE FP32***"

echo " Running the warmup..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 warmup.py ane

echo "powermetrics"
sudo rm -f ane_32.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > ane_32.txt &

echo " Running the jacobi model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 run_multigridjacobi.py $size 100 $number_exec fp32 ane

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py ane_32.txt ${size} fp32 ane
/Users/alvarocorrochano/bin/.pyenv/bin/python3 allGraphic.py ane_32.txt ${size} fp32 ane
echo " Done!"

echo " Saving vs metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 graphic16vs32.py ane_16.txt ane_32.txt ${size} ane
