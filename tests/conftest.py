from os import path
from shutil import rmtree
from pytest import fixture
from logging import getLogger

logger = getLogger(__name__)

@fixture(scope='session')
def directory():
    yield 'data/test'
    try:
        rmtree('data')
    except PermissionError:
        logger.warning('Could not remove data directory')