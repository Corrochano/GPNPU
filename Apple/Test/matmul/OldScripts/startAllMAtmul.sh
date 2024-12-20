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

if [ $# -lt 3 ]; then
    echo "Usage: $0 <param1> <param2> <param3>"
    exit 1
fi

size=$1
number_exec=$2
precision=$3

echo " *** Running CPU ***"

echo " Running the warmup..."
/Users/acorrochano/bin/pyenv/bin/python3 warmup.py cpu

echo "powermetrics"
sudo rm -f cpu.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > cpu.txt &

echo " Running the matmul model..."
# Test to launch (exec)
/Users/acorrochano/bin/pyenv/bin/python3 run_matmul.py $size $number_exec $precision cpu

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/acorrochano/bin/pyenv/bin/python3 calculatePower.py cpu.txt ${size} $precision cpu
/Users/acorrochano/bin/pyenv/bin/python3 allGraphic.py cpu.txt ${size} $precision cpu
echo " Done!"

echo " *** Running GPU ***"

echo " Running the warmup..."
/Users/acorrochano/bin/pyenv/bin/python3 warmup.py gpu

echo "powermetrics"
sudo rm -f gpu.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > gpu.txt &

echo " Running the matmul model..."
# Test to launch (exec)
/Users/acorrochano/bin/pyenv/bin/python3 run_matmul.py $size $number_exec $precision gpu

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/acorrochano/bin/pyenv/bin/python3 calculatePower.py gpu.txt ${size} $precision gpu
/Users/acorrochano/bin/pyenv/bin/python3 allGraphic.py gpu.txt ${size} $precision gpu
echo " Done!"

echo " *** Running ANE ***"

echo " Running the warmup..."
/Users/acorrochano/bin/pyenv/bin/python3 warmup.py ane

echo "powermetrics"
sudo rm -f ane.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > ane.txt &

echo " Running the matmul model..."
# Test to launch (exec)
/Users/acorrochano/bin/pyenv/bin/python3 run_matmul.py $size $number_exec $precision ane

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/acorrochano/bin/pyenv/bin/python3 calculatePower.py ane.txt ${size} $precision ane
/Users/acorrochano/bin/pyenv/bin/python3 allGraphic.py ane.txt ${size} $precision ane
echo " Done!"

