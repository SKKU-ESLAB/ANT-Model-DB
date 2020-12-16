import pymysql
import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.download import download_testdata
from tvm.contrib.utils import tempdir
from tvm.driver import tvmc
import onnx
import os
import numpy as np

import logging
logging.getLogger("").setLevel(logging.ERROR)

def recvall(sock, nbytes):
    res = []
    nread = 0
    while nread < nbytes:
        chunk = sock.recv(min(nbytes - nread, 1024))
        if not chunk:
            raise IOError("connection error")
        nread += len(chunk)
        res.append(chunk)
    return b"".join(res)

class KernelTask():
    def __init__(self, task, target, device_key):
        self.task = task
        self.target = target
        self.device_key = device_key
        self.best_config = None
        self.best_latency = None

        self.early_stopping = None
        self.record = None
        self.tuner = 'xgb'
        self.n_trial = 30

        self.measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="default"),
                runner=autotvm.RPCRunner(
                    device_key,
                    host="115.145.179.79",
                    port=9090,
                    number=5,
                    timeout=10,
                ),
            )

    def tune(self):
        # create tuner
        if self.tuner == "xgb" or self.tuner == "xgb-rank":
            tuner_obj = XGBTuner(self.task, loss_type="rank")
        elif self.tuner == "xgb_knob":
            tuner_obj = XGBTuner(selftask, loss_type="rank", feature_type="knob")
        elif self.tuner == "ga":
            tuner_obj = GATuner(self.task, pop_size=50)
        elif self.tuner == "random":
            tuner_obj = RandomTuner(self.task)
        elif self.tuner == "gridsearch":
            tuner_obj = GridSearchTuner(self.task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        def _search_best_config():
            def _callback(_, inputs, results):
                for inp, result in zip(inputs, results):
                    new_latency = result.costs[0] if result.error_no ==0 else 1e9
                    if self.record == None or self.best_latency > new_latency:
                        self.record = autotvm.record.encode(inp, result)
                        self.best_config = inp.config.to_json_dict()['entity']
                        self.best_latency = new_latency

            return _callback

        # do tuning
        task_trial = min(self.n_trial, len(self.task.config_space))
        tuner_obj.tune(
            n_trial=task_trial,
            early_stopping=self.early_stopping,
            measure_option=self.measure_option,
            callbacks=[
                autotvm.callback.progress_bar(task_trial),
                _search_best_config(),
            ],
        )
        print(self.record)

        kernel_db = KernelDB()
        kernel_db.write_task(self)
        del kernel_db


class KernelModel():
    def __init__(self, model_name, target, device_key):
        self.model_name = model_name
        self.target = target
        self.device_key = device_key
        self.latency = None
        self.graph = None
        self.params = None
        self.code = None

    def generate_model(self):
        raise NotImplementedError('')


class KernelDB():
    def __init__(self, host='115.145.178.78', port=3306, user='test', password='eslab100', db='modeldb'):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = pymysql.connect(host=host, port=port, user=user, password=password, db=db)
        self.cursor = self.db.cursor()
        
        self.device_key_list = ['xu4']
        #self.target_list = [tvm.target.Target("llvm -device=arm_cpu -mtriple=armv7l-linux-gnueabihf -mattr=+neon")]
        self.target_list = ["llvm -device=arm_cpu -mtriple=armv7l-linux-gnueabihf -mattr=+neon"]

        self.measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="default"),
                runner=autotvm.RPCRunner(
                    self.device_key_list[0],
                    host="115.145.179.79",
                    port=9090,
                    number=5,
                    timeout=10,
                ),
            )

    def __del__(self):
        self.db.close()

    def check_task(self, task):
        '''
        self.curs.execute(
                """SELECT """
                )
        if self.curs.rowcount == 0:
            return False
        else:
            return True
        '''
        raise NotImplementedError('')

    def write_task(self, kernel_task):
        sql = f"""INSERT INTO kernel_table_layer(task_name, args, target, device_key, configs, latency) 
                    VALUES ('{kernel_task.task.name}', 
                            "{str(kernel_task.task.args)}", 
                            '{kernel_task.target}', 
                            '{kernel_task.device_key}', 
                            "{kernel_task.best_config}", 
                            {kernel_task.best_latency});"""
        print(sql)
        self.cursor.execute(sql)
        self.db.commit()

    def read_task(self):
        raise NotImplementedError('')

    def delete_task(self):
        raise NotImplementedError('')

    def check_model(self):
        raise NotImplementedError('')

    def write_model(self, model_id, module_file, times):
        latency = np.mean(times)
        sql = f"""INSERT INTO kernel_table_model(device_key, target, latency, link, model_id)
                  VALUES ('{self.device_key_list[0]}',
                          '{self.target_list[0]}',
                          {latency},
                          '{module_file}',
                          {model_id});"""
        self.cursor.execute(sql)
        self.db.commit()

    def read_model(self):
        raise NotImplementedError('')

    def delete_model(self):
        raise NotImplementedError('')
    
    def find_new_model(self):

        #sql = "show full columns from model_table_prune"
        #self.cursor.execute(sql)
        #column = [i[0] for i in self.cursor.fetchall()]

        #sql = "select * from model_table_prune"
        #self.cursor.execute(sql)
        #rows = self.cursor.fetchall()

        sql = """SELECT id, onnx_link
                 FROM model_table_prune
                 WHERE id NOT IN (
                     SELECT model_id
                     FROM kernel_table_model
                 )"""
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()

        if self.cursor.rowcount == 0:
            return None
        
        return rows

    def tune_model(self, model_list):
        if model_list is None:
            return

        for model_id, onnx_link in model_list:
            model_path = download_testdata(onnx_link, str(model_id) + '.onnx', module="onnx_model")
            print(model_path)
            print(onnx_link)
            tmp = tempdir()
            log_file = tmp.relpath('tuning.log')

            mod, params = tvmc.frontends.load_model(model_path)
            tasks = tvmc.autotuner.get_tuning_tasks(mod, params, self.target_list[0])
            tvmc.autotuner.tune_tasks(
                    tasks=tasks,
                    log_file=log_file,
                    measure_option=self.measure_option,
                    tuner='xgb',
                    trials=10,
                    early_stopping=None,
                    tuning_records=None)

            graph, lib, params, dumps = tvmc.compiler.compile_model(
                    model_path, target=self.target_list[0],
                    dump_code=None)

            module_file = os.path.join('/home/modeldb/modeldb_dir/kernel_lib', str(model_id) + '.tar')
            tvmc.compiler.save_module(module_file, graph, lib, params, cross='arm-linux-gnueabihf-gcc')

            outputs, times = tvmc.runner.run_module(
                    module_file,
                    '115.145.179.79',
                    port=9090,
                    rpc_key=self.device_key_list[0],
                    device='cpu',
                    inputs_file=None,
                    fill_mode="random",
                    repeat=1,
                    profile=False)

            print(times)
            
            self.write_model(model_id, os.path.join('/downloads/kernel_lib', str(model_id) + '.tar'), times)


