import os

from voigt.app import app

if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'

print('')
print('#'*78)
print(f'Running in {os.getcwd()} on {env} environment.')

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
