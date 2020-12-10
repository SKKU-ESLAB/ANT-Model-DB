import socket
import pickle
import argparse
import struct

import tvm
from tvm import relay, autotvm
import tvm.relay.testing


def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    dtype = 'float32'
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


def main(args):
    print("Extract tasks...")
    target = tvm.target.Target('llvm -device=arm_cpu -mtriple=armv7l-linux-gnueabihf -mattr=+neon')
    mod, params, input_shape, _ = get_network(args.net, batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    print("Tuning...")
    for i, tsk in enumerate(tasks):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((args.host, args.port))
        data = pickle.dumps({'task': tsk, 'device_key': 'xu4', 'target': target, 'tuner': 'xgb'})

        client_socket.sendall(struct.pack("<i", len(data)))
        client_socket.sendall(data)
        data = client_socket.recv(1024)
        print('Received', repr(data.decode()))
        client_socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='0.0.0.0', help="The host address of the Kernel DB")
    parser.add_argument("--port", type=int, default=8003, help="The port of the Kernel DB")
    parser.add_argument("--net", type=str, default='vgg-16', help="The name of network for tuning")

    args = parser.parse_args()
    main(args)
