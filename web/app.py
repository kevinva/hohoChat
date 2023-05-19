import os
import sys
import time

dev_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "dev")
sys.path.append(dev_root)

from flask import Flask, render_template, request, jsonify
import hoho_law_ai_core as law_core

app = Flask(__name__)

# 假设这是您的初始数据
data = {'name': 'John', 'age': 25}

@app.route('/')
def index():
    return render_template('index.html', data=data)

@app.route('/query', methods=['POST'])
def query():

    law_core.main()

    time.sleep(3)

    # 获取表单提交的数据
    new_name = request.form['name']
    new_age = request.form['age']

    # 更新数据
    data['name'] = new_name
    data['age'] = new_age

    # 返回更新后的数据作为JSON响应
    return jsonify(data)

if __name__ == '__main__':
    app.run()