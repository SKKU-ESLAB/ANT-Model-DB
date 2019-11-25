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
import scipy.misc
import argparse
from time import sleep
eps = 1e-3

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='nvidia',
                    help='Model name [default: nvidia]')
parser.add_argument('--android', type=str2bool, default=False,
                    help='use_android')
parser.add_argument('--gpu', type=str2bool, default=False,
                    help='GPU or not')
parser.add_argument('--ntrial', type=int, default=200,
                    help='The number of trial')
parser.add_argument('--opt', type=int, default=0,
                    help='Optimization level')
FLAGS = parser.parse_args()
MODEL_NAME = FLAGS.model
use_android = FLAGS.android
NTRIAL = FLAGS.ntrial
OPT_LEVEL = FLAGS.opt

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = relay.Module.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


### DEVICE CONFIG ###
if FLAGS.gpu:
    target = tvm.target.create('opencl -device=mali')
    proc = "-gpu"
else:
    target = tvm.target.create('llvm -device=arm_cpu -target=armv7l-linux-gnueabihf -mattr=+neon')
    proc = "-cpu"

target=tvm.target.cuda()
device_key='1080ti'
target_host='llvm -target=arm64-linux-gnueabihf'

trial_str = '-' + str(NTRIAL)
### TUNING OPTION ###

network = MODEL_NAME + proc + trial_str + '-odroid'
print('network:', network)
log_file = "%s.log" % network
dtype='float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': NTRIAL,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        #runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        runner=autotvm.RPCRunner(
            '1080ti',
            '0.0.0.0', 9123,
            number=5, timeout=10000)
    ),
}
def conv_5x5(net, out_channels, prefix, activation=True, kernel=(5,5)):
    # 5x5 Convolution
    weight = relay.var(prefix + "_weight")
    net = relay.nn.conv2d(
        net, weight,
        strides=(1,1), padding=(0, 0),
        channels=out_channels,
        kernel_size=kernel)

    # BatchNorm
    base_name = prefix + "_bn"
    gamma = relay.var(base_name + "_gamma")
    beta = relay.var(base_name + "_beta")
    mean = relay.var(base_name + "_mean")
    var = relay.var(base_name + "_var")
    net = relay.nn.batch_norm(net, gamma=gamma, beta=beta,
                                   moving_mean=mean, moving_var=var,
                                   axis=1, epsilon=eps)[0]

    # ReLU
    if activation:
        net = relay.nn.relu(net)

    return net

def dense(net, units, prefix, activation=True, bn=True):

    # Dense
    if units == 256:
        weight1 = relay.var(prefix + "_1_weight")
        weight2 = relay.var(prefix + "_2_weight")
        net1 = relay.nn.dense(net, weight1, 128)
        net2 = relay.nn.dense(net, weight2, 128)
        net = relay.concatenate((net1, net2), 1)
    else:
        weight = relay.var(prefix + "_weight")
        net = relay.nn.dense(net, weight, units)

    # Bias add
    #bias = relay.var(prefix + "_bias")
    #net = relay.nn.bias_add(net, bias, axis=-1)

    # BatchNorm
    if bn:
        base_name = prefix + "_bn"
        gamma = relay.var(base_name + "_gamma")
        beta = relay.var(base_name + "_beta")
        mean = relay.var(base_name + "_mean")
        var = relay.var(base_name + "_var")
        net = relay.nn.batch_norm(net, gamma, beta,
                          mean, var, axis=1, epsilon=eps)[0]
    # ReLU
    if activation:
        net = relay.nn.relu(net)

    return net

def get_net(batch_size, image_shape, dtype):
    height = image_shape[1]
    width = image_shape[2]
    data_shape = (batch_size,) + image_shape
    net = relay.var("data", shape=data_shape, dtype=dtype)

    for i, dim in enumerate([24, 36, 64, 64]):
        scope = "conv" + str(i+1)
        net = conv_5x5(net, dim, scope, activation=True, kernel=(11,11))

    net = relay.transpose(net, [0,2,3,1])
    net = relay.nn.batch_flatten(net)

    for i, dim in enumerate([10]):
        fc_scope = "fc" + str(i+1)
        net = dense(net, dim, fc_scope, activation=True, bn=True)
    net = dense(net, 2, "fc5", activation=False, bn=False)

    out = net
    args = relay.analysis.free_vars(out)

    return relay.Function(args, out)

def get_workload(batch_size=1,
                 image_shape=(3, 224, 224),
                 dtype="float32",
                 **kwargs):
    net = get_net(batch_size=batch_size,
                  image_shape=image_shape,
                  dtype=dtype,
                  **kwargs)
    return create_workload(net)


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
    if os.path.exists(log_filename):
        return
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))
        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            print("***** xgb ******")
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
        done_flag = False
        while (not done_flag):
            try:
                tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])
                done_flag = True
            except:
                print("\nexception happened... wait 20 seconds.")
                sleep(20)


    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def main():

    input_shape = (1, 3, 224, 224)
    # Extract workloads from relay program
    print("Extract tasks...")
    network = MODEL_NAME
    if network is 'nvidia':
        print('get-nvidia')
        mod, params = get_workload(image_shape=input_shape[1:], batch_size=input_shape[0])
    else: 
        print('get-else')
        mod, params, input_shape, _ = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, target_host=target_host,
                                              params=params, 
                                              ops=(relay.op.nn.conv2d,relay.op.nn.dense,))
    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_option)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...(opt_level={})".format(OPT_LEVEL))
        with relay.build_config(opt_level=OPT_LEVEL):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params, target_host=target_host)
        
        # export library
        tmp = tempdir()
        if use_android:
            from tvm.contrib import ndk
            filename = "net.so"
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

        # upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9123, timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        ctx = remote.context(str(target), 0)
        module = runtime.create(graph, rlib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))

        print("Run...")
        print("set_input(\"data\")") 
        module.set_input('data', data_tvm)
        print("set_input(**params)")
        module.set_input(**params)
   
        # evaluate
        
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=50)
        prof_res = np.array(ftimer().results) * 1000    # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

if __name__ == "__main__":
    main()
