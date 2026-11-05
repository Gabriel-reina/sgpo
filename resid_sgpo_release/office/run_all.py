"""Train the fifteen method/seed combinations in independent run directories."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    for seed in range(42,47):
        for method in ['grpo','gspo','sgpo']:
            subprocess.run([sys.executable,str(ROOT/'train.py'),'--method',method,'--seed',str(seed),'--output',str(args.output/f'seed{seed}-{method}')],check=True)
if __name__=='__main__':main()
