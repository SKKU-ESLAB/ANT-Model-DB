import pymysql
import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

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
    def __init__(self, host='115.145.178.78', port=3306, user='test', password='eslab100', db='kerneldb'):
        self.db = pymysql.connect(host=host, port=port, user=user, password=password, db=db)
        self.cursor = self.db.cursor()

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
        sql = f"""INSERT INTO layers(task_name, args, target, device_key, configs, latency) 
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

    def write_model(self):
        raise NotImplementedError('')

    def read_model(self):
        raise NotImplementedError('')

    def delete_model(self):
        raise NotImplementedError('')
