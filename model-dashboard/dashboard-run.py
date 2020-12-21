from flask import Flask, \
                  send_file, send_from_directory, \
                  render_template
from read_db import make_csv, open_csv
from make_html import make_table, make_table_css, make_graph, make_combined_page, make_code_link
import pandas as pd
import tarfile
import os
app = Flask(__name__)


@app.route('/')
def first_page():
    item = "model_table_prune"
    make_csv(item)
    df = open_csv(item)
    table = make_table_css(item)
    graph = make_graph(item)
    code_link = make_code_link(item)
    with open('templates/combine.html', 'w') as f:
        f.write("{% extends 'layout.html' %}\n"\
                '{% block content0 %}Pruned models  {% endblock %}'\
                '{% block content3 %}'+ code_link +'{% endblock %}'
                '{% block content1 %}\n' + graph + '{% endblock %}\n'\
                '{% block content2 %}\n' + table + '{% endblock %}\n')

    return render_template('combine.html')

@app.route('/pruning')
def pruning():
    return first_page()


@app.route('/quantization')
def quantization():
    item = "model_table_quant"
    make_csv(item)
    df = open_csv(item)
    table = make_table_css(item)
    graph = make_graph(item)
    code_link = make_code_link(item)
    with open('templates/combine.html', 'w') as f:
        f.write("{% extends 'layout.html' %}\n"\
                '{% block content0 %}Quantized models  {% endblock %}'\
                '{% block content3 %}'+ code_link +'{% endblock %}'
                '{% block content1 %}\n' + graph + '{% endblock %}\n'\
                '{% block content2 %}\n' + table + '{% endblock %}\n')

    return render_template('combine.html')

@app.route('/kernel_layers')
def kernel_layers():
    item = "kernel_table_layer"
    make_csv(item, host="<host>")
    df = open_csv(item)
    table = make_table_css(item)
    with open('templates/combine.html', 'w') as f:
        f.write("{% extends 'layout.html' %}\n"\
                '{% block content0 %}Kernel (Layers) {% endblock %}'\
                '{% block content2 %}\n' + table + '{% endblock %}\n')
    
    return render_template('combine.html')

@app.route('/kernel_models')
def kernel_models():
    item = "kernel_table_model"
    make_csv(item, host="<host>")
    df = open_csv(item)
    table = make_table_css(item)
    with open('templates/combine.html', 'w') as f:
        f.write("{% extends 'layout.html' %}\n"\
                '{% block content0 %}Model Library {% endblock %}'\
                '{% block content2 %}\n' + table + '{% endblock %}\n')
    
    return render_template('combine.html')


@app.route('/downloads/<path:filename>')
def download(filename):
    full_path = os.path.join('modeldb_dir', filename)
    if os.path.isdir(full_path):
        with tarfile.open(full_path+'.tar', 'w:gz') as tar:
            tar.add(full_path, arcname=os.path.basename(full_path))
        return send_file(full_path+'.tar', as_attachment=True)
        
    else:
        return send_from_directory(directory='modeldb_dir', filename=filename, as_attachment=True)


@app.route('/index')
def get_index():
    return render_template('index.html',)

host_addr = "0.0.0.0"
port_num = "8001"


if __name__ == '__main__':
    app.run(host=host_addr, port=port_num, debug=True)


