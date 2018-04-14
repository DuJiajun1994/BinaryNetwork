from models.binary_connect import BinaryConnect
from data_providers.cifar10 import Cifar10


def get_data_provider(data_name):
    if data_name == 'cifar10':
        return Cifar10()
    else:
        raise Exception('data {} is not existed'.format(data_name))


def get_model(model_name):
    if model_name == 'binary_connect':
        return BinaryConnect()
    else:
        raise Exception('model {} is not existed'.format(model_name))
