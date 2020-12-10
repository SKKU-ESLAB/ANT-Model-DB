#!/usr/bin/env python3
import argparse
import struct
import socket
import pickle

from utils import KernelTask, recvall

def run():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind((args.host, args.port))

    server_socket.listen()

    print('Kernel tuning daemon ready')

    while True:
        print('Wait for request...')
        client_socket, addr = server_socket.accept()

        data_len = struct.unpack("<i", recvall(client_socket, 4))[0]
        data = recvall(client_socket, data_len)
        print('Data received (length=%d)' % (data_len))

        client_socket.sendall('Task received successfully'.encode())
        client_socket.close()

        # Tuning start
        data = pickle.loads(data)
        kernel_task = KernelTask(data['task'], data['target'], data['device_key'])
        kernel_task.tune()

    server_socket.close()

def main(args):
    run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='0.0.0.0', help="The host address of the Kernel DB")
    parser.add_argument("--port", type=int, default=8003, help="The port of the Kernel DB")

    args = parser.parse_args()
    main(args)
