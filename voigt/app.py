from .server import app
from .layout import layout
from . import callbacks

server = app.server
app.layout = layout
app.title = 'Nanotube Analysis'
