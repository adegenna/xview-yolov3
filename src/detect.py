import argparse
import time
from sys import platform

from models import *
from datasets import *
from utils.utils import *
from InputFile import *
import torch

def detect():
    """
    Main driver script for testing the YOLOv3 network.

    | **Inputs:**
    |    *args:* command line arguments used in shell call for this main driver script. args must have a inputfilename member that specifies the desired inputfile name.

    | **Outputs:**
    |    *inputs.outdir/metrics.txt:* output metrics for specified test image given by inputs.imagepath
    |    *inputs.loaddir/<inputs.imagepath>.jpg:* test image with detected bounding boxes, classes and confidence scores
    |    *inputs.loaddir/<inputs.imagepath>.tif.txt:*  text file with bounding boxes, classes and confidence scores for all detections
    """

    # Problem setup: read input file
    parser  = argparse.ArgumentParser(description='Input filename');
    parser.add_argument('inputfilename',\
                        metavar='inputfilename',type=str,\
                        help='Filename of the input file')
    args   = parser.parse_args()
    opt    = InputFile(args);
    os.system('rm -rf ' + opt.outdir)
    os.makedirs(opt.outdir, exist_ok=True)
    opt.printInputs();
    
    # Load model 1
    model      = Darknet(opt)
    checkpoint = torch.load(opt.networksavefile, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    del checkpoint

    # Set Dataloader
    classes    = load_classes(opt.class_path)  # Extracts class labels from file
    dataloader = ImageFolder(opt.imagepath, batch_size=opt.batch_size, img_size=opt.imgsize)

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    prev_time = time.time()
    detections = None
    mat_priors = scipy.io.loadmat(opt.targetspath)
    for batch_i, (img_paths, img) in enumerate(dataloader):
        print('\n', batch_i, img.shape, end=' ')

        img_ud = np.ascontiguousarray(np.flip(img, axis=1))
        img_lr = np.ascontiguousarray(np.flip(img, axis=2))

        preds = []
        length = opt.imgsize
        ni = int(math.ceil(img.shape[1] / length))  # up-down
        nj = int(math.ceil(img.shape[2] / length))  # left-right
        for i in range(ni):  # for i in range(ni - 1):
            print('row %g/%g: ' % (i, ni), end='')

            for j in range(nj):  # for j in range(nj if i==0 else nj - 1):
                print('%g ' % j, end='', flush=True)

                # forward scan
                y2 = min((i + 1) * length, img.shape[1])
                y1 = y2 - length
                x2 = min((j + 1) * length, img.shape[2])
                x1 = x2 - length

                # Get detections
                with torch.no_grad():
                    # Normal orientation
                    chip = torch.from_numpy(img[:, y1:y2, x1:x2]).unsqueeze(0)
                    pred = model(chip)
                    pred = pred[pred[:, :, 4] > opt.conf_thres]
                    if len(pred) > 0:
                        pred[:, 0] += x1
                        pred[:, 1] += y1
                        preds.append(pred.unsqueeze(0))

        if len(preds) > 0:
            detections = non_max_suppression(torch.cat(preds, 1), opt.conf_thres, opt.nms_thres, mat_priors, img)
            img_detections.extend(detections)
            imgs.extend(img_paths)

        print('Batch %d... (Done %.3fs)' % (batch_i, time.time() - prev_time))
        prev_time = time.time()

    # Bounding-box colors
    color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    if len(img_detections) == 0:
        return

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("image %g: '%s'" % (img_i, path))

        if opt.plot_flag:
            img = cv2.imread(path)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_classes = detections[:, -1].cpu().unique()
            bbox_colors = random.sample(color_list, len(unique_classes))

            # write results to .txt file
            results_path = os.path.join(opt.outdir, path.split('/')[-1])
            if os.path.isfile(results_path + '.txt'):
                os.remove(results_path + '.txt')

            results_img_path = os.path.join(opt.outdir , path.split('/')[-1])
            with open(results_path.replace('.bmp', '.tif') + '.txt', 'a') as file:
                for i in unique_classes:
                    n = (detections[:, -1].cpu() == i).sum()
                    print('%g %ss' % (n, classes[int(i)]))

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)

                    # write to file
                    xvc = xview_indices2classes(int(cls_pred))  # xview class
                    file.write(('%g %g %g %g %g %g \n') % (x1, y1, x2, y2, xvc, cls_conf * conf))

                    if opt.plot_flag:
                        # Add the bbox to the plot
                        label = '%s %.2f' % (classes[int(cls_pred)], cls_conf) if cls_conf > 0.05 else None
                        color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
                        plot_one_box([x1, y1, x2, y2], img, label=label, color=color, line_thickness=1)

            if opt.plot_flag:
                # Save generated image with detections
                cv2.imwrite(results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'), img)

    if opt.plot_flag:
        from scoring import score
        score.score(opt.outdir, '/home/adegennaro/Projects/OGA/cat/data/xview/labels/xView_train.geojson', opt.outdir)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    detect()
    torch.cuda.empty_cache()
