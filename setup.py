from setuptools import setup, find_packages

setup(
    name='eccoseas',
    version='0.1',
    author='Mike Wood',
    author_email='mike.wood@sjsu.edu',
    platforms=["any"],
    description='This package stores tools to generate regional models from ECCO state estimates',
    packages=find_packages(include=['eccoseas','eccoseas*']),
    license="BSD",
)
