from pretty_html_table import build_table
from io import StringIO

import pymysql
import csv
import pandas as pd


def make_csv(feature, host="localhost", db='modeldb'):
    conn = pymysql.connect(host=host, user="user", password="", db=db)

    curs = conn.cursor()
    column = []
    sql = "show full columns from %s" % feature
    curs.execute(sql)
    rows = curs.fetchall()
    for i in range(len(rows)):
        column.append(rows[i][0])


    sql = "select * from %s" % feature
    curs.execute(sql)
    rows = curs.fetchall()

    rows = list(rows)
    for a in range(len(rows)):
        rows[a] = list(rows[a])

    f = open('%s.csv' % feature, 'w', encoding='utf-8', newline='')

    wr = csv.writer(f)

    wr.writerow(column)

    for i in range(len(rows)):
        wr.writerow(rows[i])
    f.close()

    conn.close()

def open_csv(feature):
    f = open('%s.csv' % feature, 'r', encoding='utf-8')
    data = pd.read_csv(f)

    return data


if __name__ == "__main__":
    item = "table1"
    try:
        make_csv(item)
        data = open_csv(item)
        print(data)
    except Exception as e:
        print(e)
        print("No data.")


