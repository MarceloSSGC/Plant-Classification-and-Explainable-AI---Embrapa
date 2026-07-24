import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--a", type=int, required=True)
parser.add_argument("--b", type=int, required=True)

args = parser.parse_args()

print(args.a)
print(args.b)


