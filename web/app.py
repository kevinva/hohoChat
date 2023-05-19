from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 假设这是您的初始数据
data = {'name': 'John', 'age': 25}

@app.route('/')
def index():
    return render_template('index.html', data=data)

@app.route('/update', methods=['POST'])
def update_data():
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