from setuptools import setup

setup(
    name='photod',
    version='0.1.0',
    install_requires=[
        'tensorflow[and-cuda]>=2.15.0,<2.16.0',
        'scipy',
        'matplotlib',
        'pandas',
        'astroML'],
)
