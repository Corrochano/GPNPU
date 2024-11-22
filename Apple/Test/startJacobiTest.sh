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

echo "powermetrics"
sudo rm -f metrics.txt
sudo powermetrics -i 100 --samplers cpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info --show-process-energy > metrics.txt &
POWERMETRICS_PID=$!

echo " Running the jacobi model..."
# Test to launch (exec)
/Users/acorrochano/bin/pyenv/bin/python3 run_jacobi.py 1 100 1 fp32 ane

echo " Model done to run!"
kill $POWERMETRICS_PID

echo " Saving metrics..."
/Users/acorrochano/bin/pyenv/bin/python3 calculatePower.py metrics.txt 1000 fp32 ane
echo " Done!"

