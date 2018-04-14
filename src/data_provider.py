class DataProvider(object):
    def __init__(self, image_size, num_classes, train_size, test_size):
        self.image_size = image_size
        self.num_classes = num_classes
        self.train_size = train_size
        self.test_size = test_size

    def get_input_fn(self, phase, batch_size, is_training):
        """

        :param phase: train, val, trainval or test
        :param batch_size
        :param is_training
        """
        assert phase in ('train', 'val', 'trainval', 'test'), 'phase {} is not supported'.format(phase)

        def input_fn():
            return self._get_input(phase, batch_size, is_training)
        return input_fn

    def _get_input(self, phase, batch_size, is_training):
        raise NotImplementedError
