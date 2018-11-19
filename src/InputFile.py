import numpy as np
from pprint import pprint
import sys

# Auxiliary class for packaging input file options
class InputFile():
    def __init__(self,args=[]):
        try:
            inputfilename        = args.inputfilename
            inputfilestream      = open(inputfilename)
            self.projdir         = inputfilestream.readline().strip().split('= ')[1];
            self.datadir         = inputfilestream.readline().strip().split('= ')[1];
            self.loaddir         = inputfilestream.readline().strip().split('= ')[1];
            self.outdir          = inputfilestream.readline().strip().split('= ')[1];
            self.targetspath     = inputfilestream.readline().strip().split('= ')[1];
            self.epochs          = int(inputfilestream.readline().strip().split('= ')[1]);
            self.batchsize       = int(inputfilestream.readline().strip().split('= ')[1]);
            self.networkcfg      = inputfilestream.readline().strip().split('= ')[1];
            self.imgsize         = int(inputfilestream.readline().strip().split('= ')[1]);
            resume               = inputfilestream.readline().strip().split('= ')[1];
            self.resume          = ((resume == "True") | (resume == "true"));
            inputfilestream.close();
        except:
            sys.exit("Error: either the input file you specified does not exist, or it is formatted incorrectly.")
    def printInputs(self):
        attrs = vars(self);
        print('\n');
        print("********************* INPUTS *********************")
        print('\n'.join("%s: %s" % item for item in attrs.items()))
        print("**************************************************")
        print('\n');
