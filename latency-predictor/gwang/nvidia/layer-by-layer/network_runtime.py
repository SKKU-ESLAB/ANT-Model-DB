import os
import numpy as np
import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.relay.testing.init import create_workload

from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime 

from PIL import Image
from time import sleep
#from tvm.contrib import ndk

eps = 1e-3

def conv_3x3(net, in_channles, out_channels, prefix, activation=True):
    weight = relay.var(prefix + "_weight")
    net = relay.nn.conv2d(net, weight, padding=(1,1), channels=out_channels, kernel_size=(3,3))

    # Bias add
    bias = relay.var(prefix + "_bias")
    net = relay.nn.bias_add(net, bias)

    base_name = prefix + "_bn"
    gamma = relay.var(base_name + "_gamma")
    beta = relay.var(base_name + "_beta")
    mean = relay.var(base_name + "_mean")
    var = relay.var(base_name + "_var")
    net = relay.nn.batch_norm(net, gamma=gamma, beta=beta, moving_mean=mean, moving_var=var, axis=1, epsilon=eps)[0]

    # ReLU
    if activation:
        net = relay.nn.relu(net)

    return net
   
def get_net(batch_size, image_shape, num_classes, dtype):
    height = image_shape[1]
    width = image_shape[2]
    data_shape = (batch_size,) + image_shape
    net = relay.var("data", shape=data_shape, dtype=dtype)

    net = conv_3x3(net, 3, 64, "conv1_1")
        
    out = net
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

#target = tvm.target.cuda()
target_host = 'llvm -target=aarch64-linux-gnu'
#target = tvm.target.cuda("-model=tx2")
target = tvm.target.create('llvm -target=aarch64-linux-gnu')


network = 'sample'
log_file = 'gpu.log'
dtype = 'float32'

device_key = 'tx2'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 300,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        #runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        runner=autotvm.RPCRunner(
            device_key,  # change the device key to your key
            '0.0.0.0', 9192,
            number=5, repeat=3, timeout=100, min_repeat_ms=150)
    ),
}

def tune_tasks(tasks,
              measure_option,
              tuner='xgb',
              n_trial=300,
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
        n_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def main():
    # extract workloads from relay program
    input_shape = (1, 3, 224, 224)
    print("Extrack tasks...")
    mod, params= get_workload(image_shape = input_shape[1:], batch_size=input_shape[0])

    tasks = autotvm.task.extract_from_program(mod["main"], target=target, target_host=target_host, params=params, ops=(relay.op.nn.conv2d, relay.op.nn.dense,))

    # run tuning tasks
    print("Tuning...")

    tune_tasks(tasks, **tuning_option)

    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=1):
            graph, lib, params = relay.build_module.build(mod, target=target, params=params, target_host=target_host)

            tmp = tempdir()
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

            remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9192, timeout=10000)
            remote.upload(tmp.relpath(filename))
            rlib = remote.load_module(filename)

            ctx = remote.context(str(target), 0)
            module = runtime.create(graph, rlib, ctx)
            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))

            print("Run...")
            print("Set_input(\"data\")")
            module.set_input('data', data_tvm)
            print("Set_input(**param)")
            module.set_input(**params)

            #evaluate
            print("Evaluate inference time cost...")
            ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
            prof_res = np.array(ftimer().results) * 1000
            tmp = sorted(ftimer().results)
            print(tmp[0])
            print("Mean inference time (std dev): %.2f ms (%.2f ms)" %(np.mean(prof_res), np.std(prof_res)))

if __name__ == "__main__":
    main()

            














