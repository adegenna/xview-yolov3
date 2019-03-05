import argparse
import time
from sys import platform
from models import *
from datasets import *
from utils.utils import *
from InputFile import *

class NetworkTrainer():
    def __init__(self, model, dataloader, inputs):
        self.__inputs      = inputs;
        self.__dataloader  = dataloader;
        self.model    = model;
        self.setupCuda();
        self.loadSavedModels();

    def setupCuda(self):
        cuda          = torch.cuda.is_available()
        self.__device = torch.device('cuda:0' if cuda else 'cpu')
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if cuda:
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = True

    def loadSavedModels(self):
        if self.__inputs.resume:
            checkpoint = torch.load(self.__inputs.loaddir + 'latest.pt', map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            if torch.cuda.device_count() > 1:
                print('Using ', torch.cuda.device_count(), ' GPUs')
                self.model = nn.DataParallel(self.model)
            self.model.to(self.__device).train() # Set relevant model modules to training mode (e.g., dropout layers)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.__start_epoch = checkpoint['epoch'] + 1
            self.__best_loss   = checkpoint['best_loss']
            del checkpoint  # current, saved
        else:
            if torch.cuda.device_count() > 1:
                print('Using ', torch.cuda.device_count(), ' GPUs')
                self.model = nn.DataParallel(self.model)
            self.__start_epoch = self.__inputs.epochstart;
            self.__best_loss   = float('inf')
            self.model.to(self.__device).train()
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4, weight_decay=5e-4)

    def backpropBatchLoss(self,nBatch,epoch,imgs,targets,j):
        targets_j = targets[j * nBatch : j * nBatch + nBatch]
        imgs_j    =    imgs[j * nBatch : j * nBatch + nBatch].to(self.__device)
        nGT       = sum([len(x) for x in targets_j])
        if nGT < 1:
            flagContinue = True;
        else:
            loss = self.model(imgs_j, targets_j, requestPrecision=True,
                              weight=self.class_weights, epoch=epoch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            flagContinue = False;
        return flagContinue;

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

    def train(self):
        print('********************* NETWORK TRAINING *********************')
        print('Network architecture:')
        #modelinfo(self.model)
        t0, t1 = time.time(), time.time()
        print('%10s' * 16 % (
            'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R', 'nGT', 'TP', 'FP', 'FN', 'time'))
        class_weights      = self.__dataloader.targets_metadata.filtered_class_weights
        self.class_weights = (torch.from_numpy(class_weights).float()).to(self.__device)
        n_class            = len(self.__dataloader.targets_metadata.filtered_class_labels)
        # Main training loop
        for epoch in range(self.__inputs.epochs):
            epoch       += self.__start_epoch
            self.__ui    = -1
            self.__rloss   = defaultdict(float)  # running loss
            self.__metrics = torch.zeros(4, n_class)
            for i, (imgs, targets) in enumerate(self.__dataloader):
                n = 4  # number of pictures at a time
                for j in range(int(len(imgs) / n)):
                    flagContinue = self.backpropBatchLoss(n,epoch,imgs,targets,j);
                    if flagContinue:
                        continue;
                    self.updateMetricsLoss();
                    mean_precision = self.calculatePrecision();
                    mean_recall    = self.calculateRecall();
                    # Output metrics
                    s  = self.outputBatchMetrics(epoch,mean_precision,mean_recall,i,t1);
                    t1 = time.time()
            # Write epoch results
            with open(self.__inputs.outdir + 'results.txt', 'a') as file:
                file.write(s + '\n')
            # Update best loss
            loss_per_target = self.__rloss['loss'] / self.__rloss['nGT']
            if loss_per_target < self.__best_loss:
                self.__best_loss = loss_per_target
            # Save latest checkpoint
            checkpoint = {'epoch'    : epoch,
                          'best_loss': self.__best_loss,
                          'model'    : self.model.state_dict(),
                          'optimizer': self.optimizer.state_dict()}
            torch.save(checkpoint, self.__inputs.loaddir+'latest.pt')
            # Save best checkpoint
            if self.__best_loss == loss_per_target:
                os.system('cp ' + self.__inputs.loaddir + 'latest.pt ' + self.__inputs.loaddir + 'best.pt')
            # Save backup checkpoint
            if (epoch > 0) & (epoch % 100 == 0):
                os.system('cp ' + self.__inputs.loaddir + 'latest.pt ' + self.__inputs.loaddir + 'backup' + str(epoch) + '.pt')
        # Save final model
        dt = time.time() - t0
        print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))
        print('************************************************************')
        print('\n');
