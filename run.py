import os

from voigt.app import app

if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
    DATABASE_URL = os.environ['DATABASE_URL']
else:
    env = 'Dev'
    BASE_DIR = 'C:\\Users\\Administrator\\Desktop\\voigt'

print('')
print('#'*78)
print(f'Running in {os.getcwd()} on {env} environment.')

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
