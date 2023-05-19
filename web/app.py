import os
import sys
import time

dev_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "dev")
sys.path.append(dev_root)

from flask import Flask, render_template, request, jsonify
import hoho_law_ai_core as law_core

app = Flask(__name__)

g_chat_history = []  # hoho_todo
g_data = {"answer": ""}

@app.route('/')
def index():
    return render_template('index.html', data = g_data)


@app.route('/query', methods=['POST'])
def query():
    question = request.form['question']
    answer, history = law_core.display_answer(question, g_chat_history)

    g_data['answer'] = answer

    return jsonify(g_data)


if __name__ == '__main__':
    app.run()