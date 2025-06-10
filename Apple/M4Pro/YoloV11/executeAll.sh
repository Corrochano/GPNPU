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

number_exec=$1
model=$2

echo " ***************************"
echo " *** Running FP16 Version***"
echo " ***************************"

echo " *** Running CPU***"

echo "powermetrics"
sudo rm -f cpu_16.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > cpu_16.txt &

echo " Running the yolo model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 runYolo11.py $number_exec cpu $model

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py cpu_16.txt cpu

echo " *** Running GPU***"

echo "powermetrics"
sudo rm -f cpu_16.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > gpu_16.txt &

echo " Running the yolo model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 runYolo11.py $number_exec gpu $model

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py gpu_16.txt gpu

echo " *** Running ANE***"

echo "powermetrics"
sudo rm -f cpu_16.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > ane_16.txt &

echo " Running the yolo model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 runYolo11.py $number_exec ane $model

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py ane_16.txt ane

echo " ***************************"
echo " *** Running FP32 Version***"
echo " ***************************"

echo " *** Running CPU***"

echo "powermetrics"
sudo rm -f cpu_32.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > cpu_32.txt &

echo " Running the yolo model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 runYolo11.py $number_exec cpu $model

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py cpu_32.txt cpu

echo " *** Running GPU***"

echo "powermetrics"
sudo rm -f cpu_32.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > gpu_32.txt &

echo " Running the yolo model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 runYolo11.py $number_exec gpu $model

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py gpu_32.txt gpu

echo " *** Running ANE***"

echo "powermetrics"
sudo rm -f cpu_32.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > ane_32.txt &

echo " Running the yolo model..."
# Test to launch (exec)
/Users/alvarocorrochano/bin/.pyenv/bin/python3 runYolo11.py $number_exec ane $model

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/alvarocorrochano/bin/.pyenv/bin/python3 calculatePower.py ane_32.txt ane

