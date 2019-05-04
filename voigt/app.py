import os
from os.path import join

from .server import app
from .layout import layout
from . import callbacks

if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'

UPLOAD_FOLDER = join(BASE_DIR, 'input')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
server = app.server

app.layout = layout
