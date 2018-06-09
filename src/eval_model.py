from util import get_data_provider, get_model
import tensorflow as tf
import argparse
import time
import os


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate model')
    parser.add_argument('--model', dest='model',
                        default='binary_connect', type=str)
    parser.add_argument('--data', dest='data',
                        default='cifar10', type=str)
    parser.add_argument('--net_file', dest='net_file',
                        default='cifarnet10.txt', type=str)
    parser.add_argument('--learning_rate', dest='learning_rate',
                        default=0.01, type=float)
    parser.add_argument('--num_epochs', dest='num_epochs',
                        default=500, type=int)
    parser.add_argument('--test_interval', dest='test_interval',
                        default=8, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        default=64, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    model = get_model(args.model)
    data_provider = get_data_provider(args.data)
    net = model.load_net_from_file(os.path.join('../nets', args.net_file))
    output_dir = os.path.join('../output', args.model, args.data,'01')
    # output_dir = os.path.join('../output', args.model, args.data, time.strftime("%Y%m%d%H%M%S", time.localtime()))

    classifier = tf.estimator.Estimator(model_fn=model.get_model_fn(),
                                        model_dir=output_dir,
                                        params={
                                            'net': net,
                                            'num_classes': data_provider.num_classes,
                                            'learning_rate': args.learning_rate
                                        })
    num_iters = args.num_epochs // args.test_interval
    train_steps = args.test_interval * data_provider.train_size // args.batch_size
    for i in range(num_iters):
        '''
        classifier.train(input_fn=data_provider.get_input_fn('trainval',
                                                             batch_size=args.batch_size,
                                                             is_training=True),
                         steps=train_steps)
        metrics = classifier.evaluate(input_fn=data_provider.get_input_fn('trainval',
                                                                          batch_size=args.batch_size,
                                                                          is_training=False))
        '''
        classifier.train(input_fn=data_provider.get_input_fn('train',
                                                             batch_size=args.batch_size,
                                                             is_training=True),
                         steps=train_steps)
        metrics = classifier.evaluate(input_fn=data_provider.get_input_fn('train',
                                                                          batch_size=args.batch_size,
                                                                          is_training=False))        
        train_accuracy = metrics['accuracy']
        metrics = classifier.evaluate(input_fn=data_provider.get_input_fn('test',
                                                                          batch_size=args.batch_size,
                                                                          is_training=False))
        test_accuracy = metrics['accuracy']
        print('Epoch {}'.format(args.test_interval * (i + 1)))
        print('Train accuracy {}'.format(train_accuracy))
        print('Test accuracy {}'.format(test_accuracy))
