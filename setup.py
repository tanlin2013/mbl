from setuptools import setup


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='mbl',
    version='0.0.3',
    packages=['mbl', 'mbl.model'],
    url='https://github.com/tanlin2013/mbl',
    license='MIT',
    author='Tan Tao-Lin',
    author_email='tanlin2013@gmail.com',
    description='',
    install_requires=required
)
