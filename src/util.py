from models.binary_connect import BinaryConnect
from data_providers.cifar10 import Cifar10
from data_providers.cifar100 import Cifar100
from data_providers.svhn import Svhn


def get_data_provider(data_name):
    if data_name == 'cifar10':
        return Cifar10()
    elif data_name == 'cifar100':
    	return Cifar100()
    elif data_name == 'svhn':
    	return Svhn()
    else:
        raise Exception('data {} is not existed'.format(data_name))


def get_model(model_name):
    if model_name == 'binary_connect':
        return BinaryConnect()
    else:
        raise Exception('model {} is not existed'.format(model_name))


