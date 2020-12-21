import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
from pretty_html_table import build_table
import webbrowser
import numpy as np
import mpld3
import csv
import re
matplotlib.use('Agg')

def csv_to_df(item):
    f = open(f"./{item}.csv", 'r')
    rdr = csv.reader(f)
    parsed_csv = list(rdr)
    df = pd.DataFrame(parsed_csv[1:], columns=parsed_csv[0], dtype=str)
    return df

def make_code_link(item):
    if 'prune' in item:
        html = 'Model pruning code can be found <a href="/downloads/prune_code" traget="_blank"><b>HERE</b></a>.'
        return html

    elif 'quant' in item:
        html = 'Model quantization code can be found <a href="/downloads/quant_code" traget="_blank"><b>HERE</b></a>.'
        return html


def make_table(item):
    df = csv_to_df(item)
    html= build_table(df, 'blue_light')
    return html


def make_table_css(item):
    df = csv_to_df(item)

    if item == 'kernel_table_layer':
        df = df[["id", "task_name", "args", "target", "device_key", "configs", "latency"]]
        html=df.to_html(index=False, table_id="dtBasicExample", classes="table table-striped table-bordered table-sm",
                    escape=False, render_links=True)
        return html
    elif item == 'kernel_table_model':
        df = df[["id", "model_id", "dataset", "acc_top1", "device_key", "target", "latency", "link"]]
        df['link'] = df['link'].apply(
            lambda x: f'<button class="btn btn-outline-primary" type="button" onclick="copyToClipboard(\'<host>{x}\')">copy link</button>' ) 
        html=df.to_html(index=False, table_id="dtBasicExample", classes="table table-striped table-bordered table-sm",
                    escape=False, render_links=True)
        return html

    print(type(df['download']))
    
    df['download'] = df['download'].apply(
        # 3. copy button
        lambda x: f'<button class="btn btn-outline-primary" type="button" onclick="copyToClipboard(\'<host>{x}\')">copy link</button>' ) 
        
        # 2. a href ver.
        #lambda x: f'<a href="{x}" target="_blank" >link</a>' ) 

        # 1. download ver.
        # lambda x: f'<button class="btn btn-outline-primary" type="button" onclick="window.open(\'{x}\')">click</button>' )

    df['model_size'] = df['model_size'].apply(
            lambda x: f'{(float(x)/1000000):.1f}MB')
    df['flops'] = df['flops'].apply(
            lambda x: (float(x)/1000000))
    df = df.rename(columns={'flops':'Mflops'})

    if "prune" in item:
        df = df[["id", "dataset", "model", "prune_type", "block_size", "keep_ratio", "acc_top1", 'model_size', 'Mflops', "download"]]
    elif "quant" in item:
        df = df[["id", "dataset", "model", "quant_method", "quant_type", "acc_top1", 'model_size', 'Mflops', "download"]]
    html=df.to_html(index=False, table_id="dtBasicExample", classes="table table-striped table-bordered table-sm",
                    escape=False, render_links=True)
    
    return html


def make_graph(item):
    df = csv_to_df(item)
    fig, ax = plt.subplots(figsize=(6,5))
    N = 100

    #print(df['acc_top1'])
    model_name = df['model']
    acc = np.array([float(i) for i in df['acc_top1']])
    flops = np.array([int(i)//1000000 for i in df['flops']])
    size = np.array([int(i)//1000000 for i in df['model_size']])

    print(size)

    color = ['blue', 'green', 'red', 'orange', 'darkturquoise']
    label = ['MobileNetV2', 'MobileNetV3-Large', 'MobileNetV3-Small', 'ResNet18', 'ResNet34', 'ResNet50']

    index=[]
    for model in label:
        index.append(list(filter(lambda x: model_name[x]==model, range(len(model_name)))))

    ax.grid(color='lightgray', linestyle='--', alpha=0.5)
    if item == "model_table_prune":
        s_scale=30
    elif item == "model_table_quant":
        s_scale=20

    for i in range(len(index)):
        if index[i]==[]:
            continue
        scatter = ax.scatter(flops[index[i]],
                             acc[index[i]],
                             c=[color[i]],
                             s=np.array(size[index[i]])*s_scale,
                             cmap=plt.cm.jet,
                             label=label[i]
                             )
    plt.legend()
    plt.xlabel("MFlops", fontsize=15, labelpad=10)
    plt.ylabel("Accuracy(%)", fontsize=15, labelpad=10)

    plt.rcParams.update({'legend.labelspacing':1})
    labels = ['point {0}'.format(i + 1) for i in range(N)]
    html = mpld3.fig_to_html(fig)

    return html


def make_combined_page(item):
    table = make_table(item)
    graph = make_graph(item)
    
    front = open('frontfile.html').read()
    end = open('endfile.html').read()
    middle = '</div>\n'
    fullpage = front + graph + table + end
    with open('modeldb_page.html', 'w') as f:
        f.write(fullpage)
    return fullpage
