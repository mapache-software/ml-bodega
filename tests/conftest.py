from os import path
from shutil import rmtree
from pytest import fixture

@fixture(scope='session')
def directory():
    yield 'data/test'
    rmtree('data')

    