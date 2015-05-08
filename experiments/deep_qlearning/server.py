#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from flask import Flask, render_template_string
from flask.ext.socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

this_dir = os.path.abspath(os.path.dirname(__file__))
deepy_dir = os.path.abspath(this_dir + os.sep + ".." + os.sep + "..")
model_path = this_dir + os.sep + "models" + os.sep + "puckworld_model1.gz"

import sys; sys.path.append(deepy_dir)
from agent import DQNAgent
agent = DQNAgent(8, 5)
if os.path.exists(model_path):
    print "Load model:", model_path
    agent.load(model_path)


@app.route('/')
def index():
    return render_template_string(open(this_dir + os.sep + 'test.html').read())

@socketio.on('act', namespace='/test')
def test_action(message):
    action = agent.action(message['state'])
    emit('act', {'action': action})

@socketio.on('learn', namespace='/test')
def test_learn(message):
    agent.learn(message['state'], message['action'], message['reward'], message['next_state'])

@socketio.on('save', namespace='/test')
def test_learn(message):
    print "Save model:", model_path
    agent.save(model_path)

@socketio.on('set_epsilon', namespace='/test')
def test_set_epsilon(message):
    agent.set_epsilon(message['epsilon'])

@socketio.on('connect', namespace='/test')
def test_connect():
    print "connected"

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print "disconnected"

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--port", default=5003, type=int)
    args = ap.parse_args()
    socketio.run(app, host="0.0.0.0", port=args.port, use_reloader=False)