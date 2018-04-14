class Layer:
    def to_string(self):
        raise NotImplementedError

    @staticmethod
    def parse_string(s):
        arr = s.split('_')
        if arr[0] == 'conv':
            kernel_size = int(arr[1])
            num_outputs = int(arr[2])
            layer = Conv(kernel_size=kernel_size, num_outputs=num_outputs)
        elif arr[0] == 'pool':
            kernel_size = int(arr[1])
            stride = int(arr[2])
            layer = Pool(kernel_size=kernel_size, stride=stride)
        elif arr[0] == 'fc':
            num_outputs = int(arr[1])
            layer = FC(num_outputs=num_outputs)
        else:
            raise Exception('layer type {} is not existed'.format(arr[0]))
        return layer


class Conv(Layer):
    def __init__(self, num_outputs, kernel_size):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size

    def to_string(self):
        return 'conv_{}_{}'.format(self.kernel_size, self.num_outputs)


class Pool(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def to_string(self):
        return 'pool_{}_{}'.format(self.kernel_size, self.stride)


class FC(Layer):
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

    def to_string(self):
        return 'fc_{}'.format(self.num_outputs)
