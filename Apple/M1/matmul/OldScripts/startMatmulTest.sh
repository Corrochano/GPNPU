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

if [ $# -lt 5 ]; then
    echo "Usage: $0 <param1> <param2> <param3> <param4> <param5>"
    exit 1
fi

file_name=$1
size=$2
number_exec=$3
precision=$4
mode=$5

echo " Running the warmup..."
/Users/acorrochano/bin/pyenv/bin/python3 warmup.py $mode

echo "powermetrics"
sudo rm -f $file_name
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > $file_name &

echo " Running the jacobi model..."
# Test to launch (exec)
/Users/acorrochano/bin/pyenv/bin/python3 run_matmul.py $size $number_exec $precision $mode

echo " Model done to run!"
sudo pkill -9 powermetrics

echo " Saving metrics..."
/Users/acorrochano/bin/pyenv/bin/python3 calculatePower.py $file_name ${size} $precision $mode
echo " Done!"

