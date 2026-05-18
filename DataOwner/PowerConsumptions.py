"""NVIDIA Jetson power-consumption sampler.

Continuously polls the Jetson ``jtop`` interface and appends a CSV row per
sample containing total board power, RAM usage, and per-core CPU stats. Run
as a subprocess by :mod:`CheckPowerConsumptionCode`. If ``jtop`` reports a
failure, the watchdog is relaunched so logging resumes automatically.
"""

import os,datetime,sys,argparse,subprocess
from jtop import jtop

parser = argparse.ArgumentParser(description="Run Power Consumption Log")
parser.add_argument("--csvFile", type=str, default="PowerConsumptions.csv", help="Power Consumption File")
args = parser.parse_args()

os.makedirs("CSV",exist_ok=True)
csvFile=f"CSV/{args.csvFile}"
# with open(csvFile,"w"):pass
print("Code Starts at "+str(datetime.datetime.now())+"......")
with jtop() as jetson:
# jetson.ok() will provide the proper update frequency
    while jetson.ok():
        # Read tegra stats
        with open(csvFile, "a") as f:
            jstat=jetson.stats
            f.write(f"{datetime.datetime.now()},{jstat['Power TOT']},{jstat['time']},{jstat['RAM']:.3}")
            for i in range(1,5):
                if "CPU"+str(i) in jstat:
                    f.write(f",{jstat['CPU'+str(i)]}")
            f.write("\n")
    print("Jetson have issue jetson.ok()",jetson.ok())
    subprocess.Popen(["python3", "CheckPowerConsumptionCode.py","1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)


