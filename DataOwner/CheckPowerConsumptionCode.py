"""Watchdog that keeps the power-consumption logger alive.

Spawned in the background by :mod:`client_worker` at startup. Verifies that
``PowerConsumptions.py`` (the actual ``jtop`` sampler) is running and, if
not, relaunches it with ``nohup`` so that energy data continues to be
collected for the entire training run.
"""

import time
import psutil
import subprocess
import os,sys



if len(sys.argv)>1:time.sleep(int(sys.argv[1]))
# Define the name of the script to check and run
script_name = "PowerConsumptions.py"

# Check if the script is already running
def is_script_running(script_name):
    for process in psutil.process_iter(attrs=['pid', 'name']):
        if process.info['name'] == 'python3' and len(process.cmdline())>1 and script_name in process.cmdline()[1]:
            return True
    return False

if not is_script_running(script_name):
    # If the script is not running, start it in the background
    # script_path = os.path.abspath(script_name)
    cmd = f"nohup python3 {script_name}>>OPow.out &"
    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    # subprocess.Popen(["python3", script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
    print(f"{script_name} started in the background.")
else:
    print(f"{script_name} is already running.")

