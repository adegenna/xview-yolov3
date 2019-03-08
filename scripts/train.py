import argparse
import torch
from yolov3.utils.utils import assert_single_gpu_support
from yolov3.src.InputFile import *
from yolov3.src.models import *
from yolov3.src.datasets.datasets import ListDataset
from yolov3.src.NetworkTrainer import *
from yolov3.src.targets.Target import *

# batch_size 8: 32*17 = 544
# batch_size 4: 32*25 = 800 (1.47 vs 544) or 32*23 = 736
# batch_size 2: 32*35 = 1120 (1.40 vs 800, 2.06 cumulative)
# batch_size 1: 32*49 = 1568 (1.40 vs 1120, 2.88 cumulative)

def main():
    """
    Main driver script for training the YOLOv3 network.

    **Inputs**

    ----------
    args : command line arguments
        Command line arguments used in shell call for this main driver script. args must have a inputfilename member that specifies the desired inputfile name.

    **Outputs**

    -------
    inputs.outdir/results.txt : text file 
        output metrics for each training epoch
    inputs.loaddir/latest.pt : YOLOv3 network PyTorch save file 
        checkpoint file for latest network configuration
    inputs.loaddir/best.pt : YOLOv3 network PyTorch save file 
        checkpoint file for best current network configuration
    inputs.loaddir/backup.pt : YOLOv3 network PyTorch save file 
        checkpoint file for backup purposes
    """

    # Problem setup: read input file
    parser  = argparse.ArgumentParser(description='Input filename');
    parser.add_argument('inputfilename',\
                        metavar='inputfilename',type=str,\
                        help='Filename of the input file')
    args          = parser.parse_args()
    inputfilename = args.inputfilename
    inputs        = InputFile(inputfilename);
    inputs.printInputs();
    
    # Problem setup
    assert_single_gpu_support()
    os.makedirs(inputs.loaddir, exist_ok=True)
    targets    = Target(inputs)
    n_classes  = targets.get_number_of_filtered_classes()
    if inputs.computeboundingboxclusters:
        networkcfg         = create_yolo_architecture(inputs,n_classes,targets.clusters_wh)
        inputs.networkcfg  = networkcfg
    object_data, class_weights, image_weights, files = targets.output_data_for_listdataset()
    dataloader = ListDataset(inputs , object_data, class_weights, image_weights, files)
    model      = Darknet(inputs.networkcfg, inputs.imgsize)
    trainer    = NetworkTrainer(model, dataloader, inputs, class_weights, n_classes);

    # Start training
    trainer.train();

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
    torch.cuda.empty_cache()
