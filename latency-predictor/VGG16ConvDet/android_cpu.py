import os

import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay.testing.init import create_workload

from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

from PIL import Image
import cv2
from tvm.contrib import ndk

def conv_3x3(net, in_channels, out_channels, prefix, activation=True):
    # 3x3 Convolution
    weight = relay.var(prefix + "_weight")
    net = relay.nn.conv2d(net, weight, padding=(1, 1), channels=out_channels, kernel_size=(3, 3))

    # Bias add
    bias = relay.var(prefix + "_bias")
    net = relay.nn.bias_add(net, bias)

    # ReLU
    if activation:
        net = relay.nn.relu(net)

    return net

def dense(net, units, prefix, activation=True):
    # Dense
    weight = relay.var(prefix + "_weight")
    net = relay.nn.dense(net, weight, units)

    # Bias add
    bias = relay.var(prefix + "_bias")
    net = relay.nn.bias_add(net, bias, axis=-1)

    # ReLU
    if activation:
        net = relay.nn.relu(net)

    return net

def bbox_transform(cx, cy, w, h, post=False):
    if post:
        xmins = cx - w / 2.
        ymins = cy - h / 2.
        xmaxs = cx + w / 2.
        ymaxs = cy + h / 2.
    else:
        xmins = cx - w / relay.const(2.)
        ymins = cy - h / relay.const(2.)
        xmaxs = cx + w / relay.const(2.)
        ymaxs = cy + h / relay.const(2.)

    return xmins, ymins, xmaxs, ymaxs

def bbox_transform_inv(xmin, ymin, xmax, ymax):
    w  = xmax - xmin + relay.const(1.0)
    h  = ymax - ymin + relay.const(1.0)
    cx = xmin + relay.const(0.5) * w
    cy = ymin + relay.const(0.5) * h
    
    return cx, cy, w, h

def set_anchors(height, width):
    H, W, B = 24, 78, 9
    anchor_shapes = np.reshape(
        [np.array(
            [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
             [ 162.,  87.], [  38.,  90.], [ 258., 173.],
             [ 224., 108.], [  78., 170.], [  72.,  43.]], dtype=np.float32)] * H * W,
        (H, W, B, 2)
    )
    center_x = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, W+1)*float(width)/(W+1)]*H*B, dtype=np.float32), 
                (B, H, W)
            ),
            (1, 2, 0)
        ),
        (H, W, B, 1)
    )
    center_y = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, H+1)*float(height)/(H+1)]*W*B, dtype=np.float32),
                (B, W, H)
            ),
            (2, 1, 0)
        ),
        (H, W, B, 1)
    )
    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_shapes), axis=3),
        (-1, 4)
    )

    return anchors

def safe_exp(w):
    slope = relay.const(np.exp(1, dtype=np.float32))
    lin_bool = w > slope
    lin_region = relay.cast(lin_bool, "float32")

    lin_out = slope * w
    exp_out = relay.exp(relay.where(lin_bool, relay.zeros_like(w), w))

    out = lin_region * lin_out + (relay.const(1.) - lin_region) * exp_out
    return out

def get_net(batch_size, image_shape, num_classes, dtype):
    height = image_shape[1]
    width = image_shape[2]
    data_shape = (batch_size,) + image_shape
    net = relay.var("data", shape=data_shape, dtype=dtype)

    net = conv_3x3(net, 3, 64, "conv1_1")
    net = conv_3x3(net, 64, 64, "conv1_2")
    net = relay.nn.max_pool2d(net, pool_size=(2, 2), strides=(2, 2), ceil_mode=True)

    net = conv_3x3(net, 64, 128, "conv2_1")
    net = conv_3x3(net, 64, 128, "conv2_2")
    net = relay.nn.max_pool2d(net, pool_size=(2, 2), strides=(2, 2), ceil_mode=True)

    net = conv_3x3(net, 128, 256, "conv3_1")
    net = conv_3x3(net, 256, 256, "conv3_2")
    net = conv_3x3(net, 256, 256, "conv3_3")
    net = relay.nn.max_pool2d(net, pool_size=(2, 2), strides=(2, 2), ceil_mode=True)

    net = conv_3x3(net, 256, 512, "conv4_1")
    net = conv_3x3(net, 512, 512, "conv4_2")
    net = conv_3x3(net, 512, 512, "conv4_3")
    net = relay.nn.max_pool2d(net, pool_size=(2, 2), strides=(2, 2), ceil_mode=True)

    net = conv_3x3(net, 512, 512, "conv5_1")
    net = conv_3x3(net, 512, 512, "conv5_2")
    net = conv_3x3(net, 512, 512, "conv5_3")

    net = relay.nn.dropout(net, rate=0.5)

    net = conv_3x3(net, 512, 72, "conv6", activation=False)
    net = relay.transpose(net, (0, 2, 3, 1))

    num_class_probs = 9 * 3
    num_confidence_scores = 9
    #num_box_delta = 9 * 4
    pred_class_probs, pred_conf, pred_box_delta = relay.split(net,
            (num_class_probs, num_class_probs + num_confidence_scores),
            axis=-1)

    # Probability
    pred_class_probs = relay.reshape(pred_class_probs, (-1, 3))
    pred_class_probs = relay.nn.softmax(pred_class_probs)
    pred_class_probs = relay.reshape(pred_class_probs, (batch_size, -1, 3))

    # Confidence
    pred_conf = relay.sigmoid(pred_conf)
    pred_conf = relay.reshape(pred_conf, (batch_size, -1, 1))

    # Bbox_delta
    pred_box_delta = relay.reshape(pred_box_delta, (batch_size, -1, 4))
    delta_x, delta_y, delta_w, delta_h = relay.split(pred_box_delta, (1, 2, 3), axis=2)
    delta_x = relay.reshape(delta_x, (batch_size, -1))
    delta_y = relay.reshape(delta_y, (batch_size, -1))
    delta_w = relay.reshape(delta_w, (batch_size, -1))
    delta_h = relay.reshape(delta_h, (batch_size, -1))

    anchor_box = set_anchors(height, width)
    anchor_x = relay.Constant(tvm.nd.array(anchor_box[:, 0]))
    anchor_y = relay.Constant(tvm.nd.array(anchor_box[:, 1]))
    anchor_w = relay.Constant(tvm.nd.array(anchor_box[:, 2]))
    anchor_h = relay.Constant(tvm.nd.array(anchor_box[:, 3]))

    box_center_x = anchor_x + delta_x * anchor_w
    box_center_y = anchor_y + delta_y * anchor_h
    '''
    box_width    = anchor_w * relay.exp(delta_w)
    box_height   = anchor_h * relay.exp(delta_h)
    '''
    box_width    = anchor_w + safe_exp(delta_w)
    box_height   = anchor_h + safe_exp(delta_h)

    xmins, ymins, xmaxs, ymaxs = bbox_transform(box_center_x, box_center_y, box_width, box_height)
    xmins = relay.minimum(relay.maximum(relay.const(0.0), xmins), relay.const(width - 1.0))
    ymins = relay.minimum(relay.maximum(relay.const(0.0), ymins), relay.const(height - 1.0))
    xmaxs = relay.maximum(relay.minimum(relay.const(width - 1.0), xmaxs), relay.const(0.0))
    ymaxs = relay.maximum(relay.minimum(relay.const(height - 1.0), ymaxs), relay.const(0.0))

    det_boxes = relay.stack(bbox_transform_inv(xmins, ymins, xmaxs, ymaxs), axis=-1)

    probs = relay.multiply(pred_class_probs, pred_conf)
    det_probs = relay.max(probs, axis=2)
    det_class = relay.argmax(probs, axis=2)

    out = relay.Tuple([det_boxes, det_probs, det_class])
    args = relay.analysis.free_vars(out)
    
    return relay.Function(args, out)

def get_workload(batch_size=1,
                 num_classes=1000,
                 image_shape=(3, 224, 224),
                 dtype="float32",
                 **kwargs):
    net = get_net(batch_size=batch_size,
                  num_classes=num_classes,
                  image_shape=image_shape,
                  dtype=dtype,
                  **kwargs)
    return create_workload(net)

### DEVICE CONFIG ###
#target = tvm.target.cuda()
#target = tvm.target.create('opencl -device=mali')
target = tvm.target.create('llvm -device=arm_cpu -target=arm64-linux-android -mattr=+v8.2a,+dotprod')
quantize = True
device_key='android'
target_host = 'llvm -target=arm64-linux-android'

### TUNING OPTION ###
network = 'vgg-convdet-odroid'
if quantize:
    log_file = "cpu_quant.log"
else:
    log_file = "cpu.log"
dtype='float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 50,
    'early_stopping': 450,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func='ndk', timeout=10),
        #runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        runner=autotvm.RPCRunner(
            'android',
            '0.0.0.0', 9002,
            number=5, timeout=60)
    ),
}


# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
              measure_option,
              tuner='xgb',
              n_trial=1000,
              early_stopping=None,
              log_filename='tuning.log',
              use_transfer_learning=True,
              try_winograd=True):
    if try_winograd:
        for i in range(len(tasks)):
            try:    # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tunber_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
        
def transform_image(image):
    #image = np.array(image) - np.array([123.68, 116.78, 103.94])
    image = np.array(image) - np.array([103.939, 116.779, 123.68])
    #image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def read_params(params, data_path='VGG16ConvDet'):
    for name in params.keys():
        print("Load pre-trained weight: " + name)
        pretrained = np.load(os.path.join(data_path, name+'.npy'))
        params[name] = tvm.nd.array(pretrained)
    return

def batch_iou(boxes, box):
    lr = np.maximum(
        np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
        np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
        0
    )
    tb = np.maximum(
        np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
        np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
        0
    )
    inter = lr * tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
    return inter/union

def nms(boxes, probs, threshold):
    order = probs.argsort()[::-1]
    keep = [True] * len(order)

    for i in range(len(order)-1):
        ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > threshold:
                keep[order[j+i+1]] = False

    return keep

def filter_prediction(boxes, probs, cls_idx):
    TOP_N_DETECTION = 64
    PROB_THRESH = 0.005
    NMS_THRESH = 0.4
    if TOP_N_DETECTION < len(probs) and TOP_N_DETECTION > 0:
        order = probs.argsort()[:-TOP_N_DETECTION-1:-1]
        probs = probs[order]
        boxes = boxes[order]
        cls_idx = cls_idx[order]
    else:
        filtered_idx = np.nonzero(probs > PROB_THRESH)[0]
        probs = probs[filtered_idx]
        boxes = boxes[filtered_idx]
        cls_idx = cls_idx[filtered_idx]

    final_boxes = []
    final_probs = []
    final_cls_idx = []

    for c in range(3):
        idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
        keep = nms(boxes[idx_per_class], probs[idx_per_class], NMS_THRESH)
        for i in range(len(keep)):
            if keep[i]:
                final_boxes.append(boxes[idx_per_class[i]])
                final_probs.append(probs[idx_per_class[i]])
                final_cls_idx.append(c)

    return final_boxes, final_probs, final_cls_idx

def draw_box(im, box_list, label_list, color=(0, 255, 0), cdict=None, form='center'):
    assert form == 'center' or form == 'diagonal', \
        'bounding box format not accepted: {}.'.format(form)

    for bbox, label in zip(box_list, label_list):
        if form == 'center':
            bbox = bbox_transform(*bbox, post=True)

        xmin, ymin, xmax, ymax = [int(b) for b in bbox]

        l = label.split(':')[0]
        if cdict and l in cdict:
            c = cdict[l]
        else:
            c = color

        # draw box
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
        # draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)


def main():
    input_shape = (1, 3, 375, 1242)

    # Extract workloads from relay program
    print("Extract tasks...")
    mod, params = get_workload(image_shape=input_shape[1:], batch_size=input_shape[0])
    read_params(params)
    if quantize:
        relay_prog = relay.quantize.quantize(mod["main"], params=params)
        tasks = autotvm.task.extract_from_program(relay_prog, target=target, target_host=target_host,
                                                  params=params, ops=(relay.op.nn.conv2d,))
    else:
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, target_host=target_host,
                                                  params=params, ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_option)

    # compile kernels with history best records
    with autotvm.apply_history_best(""):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params, target_host=target_host)

        # export library
        tmp = tempdir()
        filename = "net.so"
        path = tmp.relpath(filename)
        lib.export_library(path, ndk.create_shared)

        img_name = 'sample.png'
        #image = Image.open(img_name).resize((375, 1242))
        image = cv2.resize(cv2.imread(img_name), (1242, 375))
        input_image = transform_image(image)

        synset_name = 'imagenet1000_clsid_to_human.txt'
        with open(synset_name) as f:
            synset = eval(f.read())

        # upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9002, timeout=10000)
        remote.upload(path)
        rlib = remote.load_module(filename)
        
        ctx = remote.cpu(0)
        module = runtime.create(graph, rlib, ctx)

        # Run
        print("Run...")
        #ctx = tvm.context(str(target), 0)
        #module = runtime.create(graph, lib, ctx)
        module.set_input('data', tvm.nd.array(input_image.astype('float32')))
        module.set_input(**params)

        module.run()
        '''
        det_boxes = module.get_output(0).asnumpy()
        det_probs = module.get_output(1).asnumpy()
        det_class = module.get_output(2).asnumpy()
        
        # Post-process
        PLOT_PROB_THRESH = 0.4
        CLASS_NAMES = ('car', 'pedestrian', 'cyclist')

        final_boxes, final_probs, final_class = filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx = [idx for idx in range(len(final_probs)) \
                if final_probs[idx] > PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        print(final_boxes)
        print(final_probs)
        print(final_class)

        cls2clr = {
            'car': (255, 199, 0),
            'cyclist': (0, 191, 255),
            'pedestrian': (255, 0, 191)
        }
        draw_box(
            image, final_boxes,
            [CLASS_NAMES[idx] + ': (%.2f)' % prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=cls2clr
        )

        cv2.imwrite('result.png', image)
        '''
        

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=1)
        prof_res = np.array(ftimer().results) * 1000    # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))


if __name__ == "__main__":
    main()
