from flask import Flask
from flask import render_template
from flask import request
import os

app = Flask(__name__)

from app.controller.AppController import *