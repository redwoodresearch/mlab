import subprocess
import sys

print("Pip install")
subprocess.run(["pip", "install", "einops"], stdout=sys.stdout, stderr=subprocess.STDOUT)
print("Do M A C H I N E L E A R N I N G")
subprocess.run(["python3", "days/w1d4/picture_learning2.py"], stdout=sys.stdout, stderr=subprocess.STDOUT)