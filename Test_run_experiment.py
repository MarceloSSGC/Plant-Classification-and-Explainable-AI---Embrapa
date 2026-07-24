import argparse
from time import sleep

#========================================

print(f"\n Inicio Python")

parser = argparse.ArgumentParser()

parser.add_argument("--BAND", type=int, required=True)

args = parser.parse_args()

BAND = args.BAND
print(BAND)

sleep(3)

print(BAND, BAND, BAND, "pyth")

print(f"\n Fim Python")

