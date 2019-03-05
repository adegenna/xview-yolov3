import sys
sys.path.append('../')
from collections import defaultdict
import torch.nn as nn
from utils.utils import *

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        dilation=1,
                                                        padding=pad,
                                                        bias=not bn))

            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU())

        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(module_def['stride']))  # , mode='bilinear', align_corners=True)
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [float(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_idxs)
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    # YOLO Layer 0

    def __init__(self, anchors, nC, img_dim, anchor_idxs):
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (60)
        self.bbox_attrs = 5 + nC
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        if anchor_idxs[0] == (nA * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == nA:  # 3
            stride = 16
        else:
            stride = 8

        # Build anchor grids
        nG = int(self.img_dim / stride)
        self.grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
        self.grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()
        self.scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, nA, 1, 1))

    def forward(self, p, targets=None, requestPrecision=False, weight=None, epoch=None):
        FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
        device = torch.device('cuda:0' if p.is_cuda else 'cpu')

        bs = p.shape[0]
        nG = p.shape[2]
        stride = self.img_dim / nG

        if p.is_cuda and not self.grid_x.is_cuda:
            self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
            self.anchor_w, self.anchor_h = self.anchor_w.cuda(), self.anchor_h.cuda()
            # self.scaled_anchors = self.scaled_anchors.cuda()

        # x.view(4, 650, 19, 19) -- > (4, 10, 19, 19, 65)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Get outputs
        x = torch.sigmoid(p[..., 0])  # Center x
        y = torch.sigmoid(p[..., 1])  # Center y
        w = torch.sigmoid(p[..., 2])  # Width
        h = torch.sigmoid(p[..., 3])  # Height
        width = ((w.data * 2) ** 2) * self.anchor_w
        height = ((h.data * 2) ** 2) * self.anchor_h

        # Add offset and scale with anchors (in grid space, i.e. 0-13)
        pred_boxes = FT(p[..., :4].shape)
        pred_conf = p[..., 4]  # Conf
        pred_cls = p[..., 5:]  # Class

        # Training
        if targets is not None:
            # BCEWithLogitsLoss1 = nn.BCEWithLogitsLoss(reduction='sum')  # version 0.4.1
            BCEWithLogitsLoss1 = nn.BCEWithLogitsLoss(size_average=False)  # version 0.4.0
            BCEWithLogitsLoss0 = nn.BCEWithLogitsLoss()
            # BCEWithLogitsLoss2 = nn.BCEWithLogitsLoss(weight=weight, reduction='sum')
            # MSELoss = nn.MSELoss(reduction='sum')  # version 0.4.1
            MSELoss = nn.MSELoss(size_average=False)  # version 0.4.0
            CrossEntropyLoss = nn.CrossEntropyLoss(weight=weight)

            if requestPrecision:
                gx = self.grid_x[:, :, :nG, :nG]
                gy = self.grid_y[:, :, :nG, :nG]
                pred_boxes[..., 0] = x.data + gx - width / 2
                pred_boxes[..., 1] = y.data + gy - height / 2
                pred_boxes[..., 2] = x.data + gx + width / 2
                pred_boxes[..., 3] = y.data + gy + height / 2

            tx, ty, tw, th, mask, tcls, TP, FP, FN, TC = \
                build_targets(pred_boxes, pred_conf, pred_cls, targets, self.scaled_anchors, self.nA, self.nC, nG,
                              requestPrecision)
            tcls = tcls[mask]
            if x.is_cuda:
                tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()

            # Mask outputs to ignore non-existing objects (but keep confidence predictions)
            nM = mask.sum().float()
            nGT = sum([len(x) for x in targets])
            if nM > 0:
                # wC = weight[torch.argmax(tcls, 1)]  # weight class
                # wC /= sum(wC)
                lx = 2 * MSELoss(x[mask], tx[mask])
                ly = 2 * MSELoss(y[mask], ty[mask])
                lw = 4 * MSELoss(w[mask], tw[mask])
                lh = 4 * MSELoss(h[mask], th[mask])
                lconf = 1.5 * BCEWithLogitsLoss1(pred_conf[mask], mask[mask].float())

                lcls = nM * CrossEntropyLoss(pred_cls[mask], torch.argmax(tcls, 1))  # * min(epoch*.01 + 0.125, 1)
                # lcls = BCEWithLogitsLoss2(pred_cls[mask], tcls.float())
            else:
                lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

            lconf += nM * BCEWithLogitsLoss0(pred_conf[~mask], mask[~mask].float())

            loss = lx + ly + lw + lh + lconf + lcls
            i = torch.sigmoid(pred_conf[~mask]) > 0.999
            FPe = torch.zeros(self.nC)
            if i.sum() > 0:
                FP_classes = torch.argmax(pred_cls[~mask][i], 1)
                for c in FP_classes:
                    FPe[c] += 1

            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
                   nGT, TP, FP, FPe, FN, TC

        else:
            pred_boxes[..., 0] = x.data + self.grid_x
            pred_boxes[..., 1] = y.data + self.grid_y
            pred_boxes[..., 2] = width
            pred_boxes[..., 3] = height

            # If not in training phase return predictions
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride,
                                torch.sigmoid(pred_conf.view(bs, -1, 1)), pred_cls.view(bs, -1, self.nC)), -1)
            return output.data


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, inputs):
        super(Darknet, self).__init__()        
        try:
            self.module_defs = parse_model_config(inputs.networkcfg)
            self.nC          = int(self.module_defs[-1]['classes'])
        except:
            sys.exit('Loading YOLOv3 config file failed...')
        self.module_defs[0]['height'] = inputs.imgsize
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size   = inputs.imgsize
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'nGT', 'TP', 'FP', 'FPe', 'FN', 'TC']

    def forward(self, x, targets=None, requestPrecision=False, weight=None, epoch=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []

        #print("x_dims: " + str(x.size()))
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                #print(str(i) + " conv/upsample")
                x = module(x)
            elif module_def['type'] == 'route':
                #print(str(i) + " route")
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                #print(str(i) + " shortcut")
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                #print(str(i) + " yolo")
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets, requestPrecision, weight, epoch)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            #print("x_dims: " + str(x.size()))
            layer_outputs.append(x)

        if is_training:
            self.losses['nGT'] /= 3
            self.losses['TC'] /= 3
            metrics = torch.zeros(4, self.nC)  # TP, FP, FN, target_count

            ui = np.unique(self.losses['TC'])[1:]
            for i in ui:
                j = self.losses['TC'] == float(i)
                metrics[0, i] = (self.losses['TP'][j] > 0).sum().float()  # TP
                metrics[1, i] = (self.losses['FP'][j] > 0).sum().float()  # FP
                metrics[2, i] = (self.losses['FN'][j] == 3).sum().float()  # FN
            metrics[3] = metrics.sum(0)
            metrics[1] += self.losses['FPe']

            self.losses['TP'] = metrics[0].sum()
            self.losses['FP'] = metrics[1].sum()
            self.losses['FN'] = metrics[2].sum()
            self.losses['TC'] = 0
            self.losses['metrics'] = metrics

        return sum(output) if is_training else torch.cat(output, 1)


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    lines       = read_config_file_to_string_list(path)
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def create_yolo_architecture(inputs,n_classes,anchor_coordinates):
    """Creates a yolo-v3 layer configuration file from desired options

    **Inputs**

    ----------
    inputs : InputFile object 
        Specifies some necessary user options.
    n_classes : int
        Specifies number of classes in the dataset
    anchor_coordinates : list<double>
        List of doubles of form [x1,y1,x2,y2, ... , xN,yN] where N = number of anchors and (xi,yi) are the i'th anchor coordinates.

    **Outputs**

    ----------
    output_config_file_path : string
        Absolute filepath of the network config file created by this function
    """
    print('Creating custom YOLOv3 architecture from desired specifications...')
    output_config_file_path = '/'.join(inputs.networkcfg.split('/')[0:-1]) + '/yolov3_custom.cfg'
    create_yolo_config_file(inputs.networkcfg,\
                            output_config_file_path,\
                            inputs.boundingboxclusters,\
                            n_classes,\
                            anchor_coordinates)
    return output_config_file_path

def create_yolo_config_file(template_file_path,output_config_file_path,n_anchors,n_classes,anchor_coordinates):
    """Creates a yolo-v3 layer configuration file from desired options"""
    try:
        assert(len(anchor_coordinates) == 2*n_anchors)
    except:
        sys.exit('Error: length of anchor_coordinates must equal 2*n_anchors')
    lines                       = read_config_file_to_string_list(template_file_path)
    num_yolo_filters,n_anchors  = calculate_number_of_yolo_filters(n_anchors,n_classes)
    anchors_per_yolo_layer      = n_anchors // 3
    lineidx = 0
    for line in lines:
        if line.startswith('NUM_YOLO_FILTERS'):
            line           = 'filters = '
            line          += str(num_yolo_filters)
        elif line.startswith('YOLO_MASK_LARGE'):
            line  = 'mask = '
            mask  = np.arange(2*anchors_per_yolo_layer,n_anchors)
            line += str(list(mask))[1:-1]
        elif line.startswith('YOLO_MASK_MEDIUM'):
            line  = 'mask = '
            mask  = np.arange(1*anchors_per_yolo_layer,2*anchors_per_yolo_layer)
            line += str(list(mask))[1:-1]
        elif line.startswith('YOLO_MASK_SMALL'):
            line  = 'mask = '
            mask  = np.arange(0,1*anchors_per_yolo_layer)
            line += str(list(mask))[1:-1]
        elif line.startswith('YOLO_ANCHORS'):
            line  = 'anchors = '
            line += str(list(anchor_coordinates))[1:-1]
        elif line.startswith('CLASSES'):
            line  = 'classes = '
            line += str(n_classes)
        elif line.startswith('NUM_ANCHORS'):
            line  = 'num = '
            line += str(n_anchors)
        lines[lineidx] = line
        lineidx += 1
    write_string_list_to_config_file(lines,output_config_file_path)

def read_yolo_config_file_anchors(cfg_path):
    """Reads the anchor coordinates from a specified YOLO configuration file"""
    lines  = read_config_file_to_string_list(cfg_path)
    for line in lines:
        if line.startswith('anchors = '):
            anchors = line[10:]
            anchors = np.array(anchors.split(',')).astype(float)
            break
    return anchors
    
def read_config_file_to_string_list(path):
    file  = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    return lines

def write_string_list_to_config_file(lines,path):
    with open(path, 'w') as f:
        for item in lines:
            f.write("%s\n" % item)

def calculate_number_of_yolo_filters(n_anchors,n_classes):
    anchorsmod3    = n_anchors % 3
    try:
        assert(anchorsmod3 == 0)
    except:
        print('Number of anchors must be a multiple of 3 in the YOLO architecture. Therefore, the number of anchors is being changed from ' + str(n_anchors) + ' to ' + str(n_anchors - mod3))
        n_anchors -= mod3
    num_yolo_filters = (n_classes + 5)*(n_anchors // 3)
    return num_yolo_filters,n_anchors


class ConvNetb(nn.Module):
    def __init__(self, num_classes=60):
        super(ConvNetb, self).__init__()
        n = 64  # initial convolution size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(n, n * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 2),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(n * 2, n * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 4),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(n * 4, n * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 8),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(n * 8, n * 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 16),
            nn.LeakyReLU())

        self.fully_conv = nn.Conv2d(n * 16, 60, kernel_size=4, stride=1, padding=0, bias=True)

    def forward(self, x):  # 500 x 1 x 64 x 64
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fully_conv(x)
        return x.squeeze()  # 500 x 60
