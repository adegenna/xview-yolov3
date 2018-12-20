import argparse
import time
from sys import platform
import sys
sys.path.insert(0,'../')
from models import *
from utils.datasets import *
from utils.utils import *
from InputFile import *
from NetworkTrainer import *
import torch

# batch_size 8: 32*17 = 544
# batch_size 4: 32*25 = 800 (1.47 vs 544) or 32*23 = 736
# batch_size 2: 32*35 = 1120 (1.40 vs 800, 2.06 cumulative)
# batch_size 1: 32*49 = 1568 (1.40 vs 1120, 2.88 cumulative)

def main(inputs):
    # Problem setup: read input file
    parser  = argparse.ArgumentParser(description='Input filename');
    parser.add_argument('inputfilename',\
                        metavar='inputfilename',type=str,\
                        help='Filename of the input file')
    args   = parser.parse_args()
    inputs = InputFile(args);
    inputs.printInputs();
    # Problem setup
    os.makedirs(inputs.loaddir, exist_ok=True)
    model      = Darknet(inputs.networkcfg, inputs.imgsize)
    modelinfo(model)
    dataloader = ListDataset(inputs)
    trainer    = NetworkTrainer(model, dataloader, inputs);
    # Start training
    trainer.train();

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(inputs)
    torch.cuda.empty_cache()
