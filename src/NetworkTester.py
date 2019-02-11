import argparse
import time
from sys import platform
from models import *
from datasets import *
from utils.utils import *
from InputFile import *

class NetworkTester():
    """
    Class for handling testing and assessing the performance of a trained YOLOv3 model.
    | **Inputs:**
    |    *model:* trained YOLOv3 network (PyTorch .pt file).
    |    *dataloader:* dataloader object (usually an instantiation of the ImageFolder class)
    |    *inputs:* input file with various user-specified options
    """
    def __init__(self, model, dataloader, inputs):
        self.__inputs      = inputs;
        self.__dataloader  = dataloader;
        self.model         = model;
        self.setupCuda();
        self.loadSavedModels();
        self.loadClasses()

    def setupCuda(self):
        """
        Basic method to setup GPU/cuda support, if available 
        """
        cuda          = torch.cuda.is_available()
        self.__device = torch.device('cuda:0' if cuda else 'cpu')
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if cuda:
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
        
    def loadSavedModels(self):
        """
        Method to load a saved YOLOv3 model from a PyTorch (.pt) file.
        """
        checkpoint = torch.load(self.__inputs.networksavefile, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])

    def loadClasses(self):
        """
        Method to load class names from specified path in user-input file. 
        """
        self.__classes = load_classes(self.__inputs.class_path)

    def updateMetricsLoss(self):
        self.__ui += 1
        self.__metrics += self.model.losses['metrics']
        for key, val in self.model.losses.items():
            self.__rloss[key] = (self.__rloss[key] * self.__ui + val) / (self.__ui + 1)

    def calculatePrecision(self):
        precision = self.__metrics[0] / (self.__metrics[0] + self.__metrics[1] + 1e-16)
        k = (self.__metrics[0] + self.__metrics[1]) > 0
        if k.sum() > 0:
            mean_precision = precision[k].mean()
        else:
            mean_precision = 0
        return mean_precision;

    def calculateRecall(self):
        recall = self.__metrics[0] / (self.__metrics[0] + self.__metrics[2] + 1e-16)
        k      = (self.__metrics[0] + self.__metrics[2]) > 0
        if k.sum() > 0:
            mean_recall = recall[k].mean()
        else:
            mean_recall = 0
        return mean_recall;

    def outputBatchMetrics(self,epoch,mean_precision,mean_recall,i,t1):
        s = ('%10s%10s' + '%10.3g' * 14) % (
            '%g/%g' % (epoch, self.__inputs.epochs - 1), '%g/%g' % (i, len(self.__dataloader) - 1), self.__rloss['x'],
            self.__rloss['y'], self.__rloss['w'], self.__rloss['h'], self.__rloss['conf'], self.__rloss['cls'],
            self.__rloss['loss'], mean_precision, mean_recall, self.model.losses['nGT'], self.model.losses['TP'],
            self.model.losses['FP'], self.model.losses['FN'], time.time() - t1)
        print(s)
        return s;

    def detect(self):
        """Method to compute object detections over testing dataset"""

        print('********************* NETWORK TESTING *********************')
        imgs           = []  # Stores image paths
        img_detections = []  # Stores detections for each image index
        prev_time      = time.time()
        detections     = None
        for batch_i, (img_paths, img) in enumerate(self.__dataloader):
            print('\n', batch_i, img.shape, end=' ')

            img_ud = np.ascontiguousarray(np.flip(img, axis=1))
            img_lr = np.ascontiguousarray(np.flip(img, axis=2))

            preds = []
            length = self.__inputs.imgsize
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
                        pred = self.model(chip)
                        pred = pred[pred[:, :, 4] > self.__inputs.conf_thres]
                        if len(pred) > 0:
                            pred[:, 0] += x1
                            pred[:, 1] += y1
                            preds.append(pred.unsqueeze(0))

            if len(preds) > 0:
                detections = non_max_suppression(torch.cat(preds, 1), self.__inputs.conf_thres, self.__inputs.nms_thres, opt=self.__inputs, img=img)
                img_detections.extend(detections)
                imgs.extend(img_paths)

            print('Batch %d... (Done %.3fs)' % (batch_i, time.time() - prev_time))
            prev_time = time.time()

            self.__img_detections = img_detections
            self.__imgs           = imgs

    def plotDetection(self):
        """Method to plot and display all detected objects in the testing dataset"""

        # Bounding-box colors
        color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(self.__classes))]
        if len(self.__img_detections) == 0:
            return
        
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(self.__imgs, self.__img_detections)):
            print("image %g: '%s'" % (img_i, path))

            if self.__inputs.plot_flag:
                img = cv2.imread(path)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                unique_classes = detections[:, -1].cpu().unique()
                bbox_colors = random.sample(color_list, len(unique_classes))

                # write results to .txt file
                results_path = os.path.join(self.__inputs.outdir, path.split('/')[-1])
                if os.path.isfile(results_path + '.txt'):
                    os.remove(results_path + '.txt')

                results_img_path = os.path.join(self.__inputs.outdir , path.split('/')[-1])
                with open(results_path.replace('.bmp', '.tif') + '.txt', 'a') as file:
                    for i in unique_classes:
                        n = (detections[:, -1].cpu() == i).sum()
                        print('%g %ss' % (n, self.__classes[int(i)]))

                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)

                        # write to file
                        xvc = xview_indices2classes(int(cls_pred))  # xview class
                        file.write(('%g %g %g %g %g %g \n') % (x1, y1, x2, y2, xvc, cls_conf * conf))

                        if self.__inputs.plot_flag:
                            # Add the bbox to the plot
                            label = '%s %.2f' % (self.__classes[int(cls_pred)], cls_conf) if cls_conf > self.__inputs.cls_thres else None
                            color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
                            plot_one_box([x1, y1, x2, y2], img, label=label, color=color, line_thickness=1)

                if self.__inputs.plot_flag:
                    # Save generated image with detections
                    cv2.imwrite(results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'), img)

