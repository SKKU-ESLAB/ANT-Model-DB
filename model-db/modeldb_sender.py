import pymysql
import pandas as pd
import string
import random
import argparse
import os
import paramiko
from scp import SCPClient, SCPException
import csv

def record_db(metadata):
    table_name = 'model_table_prune'
    db_name = "modeldb"
    conn = pymysql.connect(host="host", 
            port="mysql" , 
            user="modeldb", 
            password="", 
            db=db_name)
    curs = conn.cursor()
    
    column_list = []
    t_cur = conn.cursor(pymysql.cursors.DictCursor)
    t_cur.execute(f'select * from information_schema.columns '\
                  f'where table_schema = "{db_name}" and table_name="{table_name}"')
    for column in t_cur:
        column_list.append(column['COLUMN_NAME'])

    str1 = f"INSERT INTO {table_name} ("
    str2 = ") VALUES('"
    str3 = "');"
    column_list = [key for key in column_list if key in metadata]
    column_str = ", ".join(column_list)
            
    value_list = [metadata[key] for key in column_list if key in metadata]
    value_str = "', '".join(value_list)
    
    sql = str1 + column_str + str2 + value_str + str3
    #print(sql)

    curs.execute(sql)
    conn.commit()
    conn.close()


def show_table():
    table_name = 'model_table_prune'
    db_name = "modeldb"
    conn = pymysql.connect(host="host",
                port="mysql",
                user="modeldb",
                password="",
                db=db_name)
    curs = conn.cursor()
    sql = "show full columns from %s" % table_name
    curs.execute(sql)
    rows = curs.fetchall()
    columns = []
    for i in range(len(rows)):
        columns.append(rows[i][0])


    sql = "select * from %s" % table_name
    curs.execute(sql)
    rows = curs.fetchall()
    rows = list(rows)
    for a in range(len(rows)):
        rows[a] = list(rows[a])
    df = pd.DataFrame(rows, columns=columns)
    print(df)


def parse_log(path):
    with open(os.path.join(path, 'train_log.txt')) as f:
        log = f.readlines()
    temp={}
    # argument
    for i in log:
        if 'Epoch start' in i:
            break
        temp[i.split(':')[0].strip()] = i.split(':')[1].split('\n')[0].strip()
    # accuracy
    for i in log:
        if 'Best test accuracy' in i:
            temp['acc_top1'] = i.split(' ')[-1].split('\n')[0] #round(float(i.split(' ')[-1].split('\n')[0]), 2)
    # flops, size (heuristic..)
    if temp['model']=='mobilenetv3-large-1.0':
        temp['model'] = 'MobileNetV3-Large'
        temp['flops'] = str(219800000  * (1-float(temp['sparsity'])))
        temp['model_size'] = str(5481000 * (1-float(temp['sparsity'])))
    path2 = path.split('/')[-2]
    temp['download']='/downloads/' + os.path.join(path2, 'removed_models.pth')
    temp['keep_ratio'] = '%s%%'%int((1-float(temp['sparsity']))*100)
    temp['onnx_link'] ='http://<base_path>' + os.path.join(path2, 'removed_models.onnx')

    
    return temp


def make_csv(table):
    conn = pymysql.connect(host="host", port="mysql" , user="modeldb", password="", db="modeldb")
    curs = conn.cursor()
    column = []
    sql = "show full columns from %s" % table
    curs.execute(sql)
    rows = curs.fetchall()
    for i in range(len(rows)):
        column.append(rows[i][0])


    sql = "select * from %s" % table
    curs.execute(sql)
    rows = curs.fetchall()

    rows = list(rows)
    for a in range(len(rows)):
        rows[a] = list(rows[a])

    f = open('%s.csv' % table, 'w', encoding='utf-8', newline='')

    wr = csv.writer(f)

    wr.writerow(column)

    for i in range(len(rows)):
        wr.writerow(rows[i])
    f.close()

    conn.close()



def open_csv(table):
    f = open('%s.csv' % table, 'r', encoding='utf-8')
    data = pd.read_csv(f)
    return data



class SSHManager:
    """
    usage:
        >>> import SSHManager
        >>> ssh_manager = SSHManager()
        >>> ssh_manager.create_ssh_client(hostname, username, password
        >>> ssh_manager.get_file("/path/to/remote_path", "/path/to/local_path")
        ...
        >>> ssh_manager.close_ssh_client()
    """
    def __init__(self):
        self.ssh_client = None

    def create_ssh_client(self, hostname, port, username, password):
        """Create SSH client session to remote server"""
        if self.ssh_client is None:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(hostname,
                                    port=port,
                                    username=username, 
                                    password=password, 
                                    allow_agent=False)
        else:
            print("SSH client session exist.")

    def close_ssh_client(self):
        """Close SSH client session"""
        self.ssh_client.close()

    def send_file(self, local_path, remote_path):
        """Send a single file to remote path"""
        with SCPClient(self.ssh_client.get_transport()) as scp:
            scp.put(local_path, remote_path=remote_path, recursive=True, preserve_times=True)


    def send_command(self, command):
        """Send a single command"""
        stdin, stdout, stderr = self.ssh_client.exec_command(command)
        return stdout.readlines()


def send(local_path, remote_path):
    ssh_manager = SSHManager()
    ssh_manager.create_ssh_client("host", 22, "modeldb", "modeldb")
    ssh_manager.send_file(local_path, remote_path)
    ssh_manager.close_ssh_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', type=str, default='model_table_prune')
    parser.add_argument('--path', type=str, help="directory to put in DB")
    parser.add_argument('--save_as_csv', action='store_true')
    args = parser.parse_args()
    
    assert args.path
    print('Record into DB')
    metadata = parse_log(args.path)
    record_db(metadata)
    show_table()
    
    if args.save_as_csv:
        make_csv(args.table)
        print("CSV file saved!")

    print('Sending files ... ')
    source_abs= os.path.abspath(args.path)
    send(source_abs, '/home/modeldb/modeldb_dir')
    print('Done!')

