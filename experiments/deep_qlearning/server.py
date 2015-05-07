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

import sys; sys.path.append(deepy_dir)
from agent import RLAgent
agent = RLAgent(8, 4)

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