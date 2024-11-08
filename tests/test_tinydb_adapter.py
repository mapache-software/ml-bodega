from pytest import fixture, raises
from mlbodega.schemas import Experiment, Model, Metric
from mlbodega.schemas import Transaction, Criterion, Optimizer, Iteration, Dataset
from mlbodega.tinydb.experiments import Experiments
from mlbodega.tinydb.models import Models

@fixture
def experiments(directory):
    return Experiments(directory)

def test_experiments(experiments: Experiments):
    experiment = experiments.create('test')
    assert experiment.name == 'test'
    experiment = experiments.read('test')
    assert experiment.name == 'test'
    assert experiments.read('test2') is None
    assert len(experiments.list()) == 1

    experiment = experiments.create('test2')
    assert experiment.name == 'test2'
    assert len(experiments.list()) == 2

    experiments.delete('test')
    experiment.name = 'test'
    experiments.update(experiment)

    experiment = experiments.read('test')
    assert experiment.name == 'test'
    with raises(ValueError):
        experiments.create('test')

@fixture
def models(directory):
    return Models(directory, Experiment(id='1', name='test'))

def test_models(models: Models):
    model = Model(id='1', hash='1', name='MLP', parameters={'hidden_units': 128}, epochs=10)
    models.put(model)
    assert models.get('1').name == 'MLP'
    assert len(models.list()) == 1

    model = Model(id='2', hash='2', name='CNN', parameters={'filters': 64}, epochs=0)
    models.put(model)
    assert models.get('2').name == 'CNN'
    assert len(models.list()) == 2
    model.epochs = 5
    models.put(model)
    assert models.get('2').epochs == 5
    assert len(models.list()) == 2
    models.remove(model)
    assert len(models.list()) == 1

def test_metrics(models: Models):
    model = Model(id='1', hash='1', name='MLP', parameters={'hidden_units': 128}, epochs=10)
    models.put(model)
    models.metrics.add(Metric(name='accuracy', value=0.9, batch=10, epoch=10, phase='train'), model)
    models.metrics.add(Metric(name='accuracy', value=0.8, batch=10, epoch=10, phase='val'), model)
    models.metrics.add(Metric(name='loss', value=0.1, batch=10, epoch=10, phase='train'), model)

    metrics = models.metrics.list(model)
    assert len(metrics) == 3

def test_transactions(models: Models):
    model = Model(id='1', hash='1', name='MLP', parameters={'hidden_units': 128}, epochs=10)
    models.put(model)
    models.transactions.put(Transaction(
        hash='123',
        epochs=(12, 15),
        start='2021-01-01T00:00:00',
        end='2021-01-01T00:00:20',
        criterion=Criterion(
            hash='1234',
            name='CrossEntropy',
            parameters={}
        ),
        optimizer=Optimizer(
            hash='123',
            name='Adam',
            parameters={'lr': 0.001}
        ),
        iterations=[Iteration(
            phase='train',
            dataset=Dataset(
                hash='123',
                name='MNIST',
                parameters={'train': True}
            ),
            parameters={'batch_size': 32}
        )]
    ), model)

    transactions = models.transactions.list(model)
    assert len(transactions) == 1

    models.transactions.put(Transaction(
        hash='123',
        epochs=(12, 20),
        start='2021-01-01T00:00:00',
        end='2021-01-01T00:00:20',
        criterion=Criterion(
            hash='1234',
            name='CrossEntropy',
            parameters={}
        ),
        optimizer=Optimizer(
            hash='123',
            name='Adam',
            parameters={'lr': 0.001}
        ),
        iterations=[Iteration(
            phase='train',
            dataset=Dataset(
                hash='123',
                name='MNIST',
                parameters={'train': True}
            ),
            parameters={'batch_size': 32}
        )]
    ), model)

    transactions = models.transactions.list(model)
    assert len(transactions) == 1