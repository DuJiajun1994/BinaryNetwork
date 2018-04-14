from data_provider import DataProvider
import tensorflow as tf
import os


class Cifar10(DataProvider):
    def __init__(self):
        # origin image size is 32 * 32, cropped to 28 * 28
        super().__init__(image_size=28, num_classes=10, train_size=50000, test_size=10000)
        self._data_path = '../data/cifar-10-batches-bin'
        self._num_preprocess_threads = 1

    def _get_input(self, phase, batch_size, is_training):
        if phase == 'test':
            filename = [os.path.join(self._data_path, 'test_batch.bin')]
        else:
            if phase == 'train':
                batch_ids = [1, 2, 3, 4]
            elif phase == 'val':
                batch_ids = [5]
            elif phase == 'trainval':
                batch_ids = [1, 2, 3, 4, 5]
            filename = [os.path.join(self._data_path, 'data_batch_{}.bin'.format(batch_id))
                        for batch_id in batch_ids]

        if is_training:
            num_epochs = None
        else:
            num_epochs = 1
        filename_queue = tf.train.string_input_producer(filename, num_epochs=num_epochs)

        images, labels = self._read_cifar10(filename_queue)

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

    def _read_cifar10(self, filename_queue):
        image_size = 32
        label_bytes = 1  # 2 for CIFAR-100
        num_channels = 3
        image_bytes = image_size * image_size * num_channels
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue.  No
        # header or footer in the CIFAR-10 format, so we leave header_bytes
        # and footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8->int32.
        labels = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
          tf.strided_slice(record_bytes, [label_bytes],
                           [label_bytes + image_bytes]),
          [num_channels, image_size, image_size])
        # Convert from [depth, height, width] to [height, width, depth].
        images = tf.transpose(depth_major, [1, 2, 0])
        images = tf.cast(images, tf.float32)
        # Set the shapes of tensors.
        images.set_shape([image_size, image_size, 3])
        labels.set_shape([1])
        return images, labels
