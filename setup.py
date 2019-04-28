from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [str.strip(fn) for fn in f.read().split('\n') if fn != '.']

setup(
    name='voigt',
    version='1.0.0',
    author='Matthew Chatham',
    author_email='matthew@matthewchatham.com',
    description='UpWork project for John Bulmer working on nanotubes',
    packages=find_packages(),
    install_requires=requirements,
)
