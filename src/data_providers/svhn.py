from data_provider import DataProvider
import tensorflow as tf
import os


class Svhn(DataProvider):
    def __init__(self):
        # origin image size is 32 * 32, cropped to 28 * 28
        super().__init__(image_size=28, num_classes=10, train_size=73257, test_size=26032)
        self._data_path = '../data/svhn'
        self._num_preprocess_threads = 1

    def _get_input(self, phase, batch_size, is_training):
        if phase == 'test':
            dataname = [os.path.join(self._data_path, 'test_data.bin')]
            labelname = [os.path.join(self._data_path, 'test_label.bin')]
        else:
            dataname = [os.path.join(self._data_path, 'train_data.bin')]
            labelname = [os.path.join(self._data_path, 'train_label.bin')]

        if is_training:
            num_epochs = None
        else:
            num_epochs = 1
        dataname_queue = tf.train.string_input_producer(dataname, num_epochs=num_epochs)
        labelname_queue = tf.train.string_input_producer(labelname, num_epochs=num_epochs)

        images = self._read_svhn_data(dataname_queue)
        labels = self._read_svhn_label(labelname_queue)

        if is_training:
            images = tf.random_crop(images, [self.image_size, self.image_size, 3])
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_brightness(images, max_delta=63)
            images = tf.image.random_contrast(images, lower=0.2, upper=1.8)
        else:
            images = tf.image.resize_image_with_crop_or_pad(images, self.image_size, self.image_size)

        images = tf.image.per_image_standardization(images)

        if is_training:
            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(self.train_size *
                                     min_fraction_of_examples_in_queue)
            images, labels = tf.train.shuffle_batch([images, labels],
                                                    batch_size=batch_size,
                                                    num_threads=self._num_preprocess_threads,
                                                    capacity=min_queue_examples + 3 * batch_size,
                                                    min_after_dequeue=min_queue_examples)
        else:
            images, labels = tf.train.batch([images, labels],
                                            batch_size=batch_size,
                                            num_threads=self._num_preprocess_threads)
        return images, labels

    def _read_svhn_data(self, filename_queue):
        image_size = 32
        num_channels = 3

        image_bytes = image_size * image_size * num_channels

        reader = tf.FixedLengthRecordReader(record_bytes=image_bytes)
        key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        depth_major = tf.reshape(
          tf.strided_slice(record_bytes, [0],[image_bytes]),[num_channels, image_size, image_size])
        # Convert from [depth, height, width] to [height, width, depth].
        images = tf.transpose(depth_major, [1, 2, 0])
        images = tf.cast(images, tf.float32)
        # Set the shapes of tensors.
        images.set_shape([image_size, image_size, 3])
        return images

    def _read_svhn_label(self, filename_queue):
        label_bytes = 1  

        record_bytes = label_bytes

        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8->int32.
        labels = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

        labels.set_shape([1])
        return labels        
