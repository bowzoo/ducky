import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)



def preprocess(data_dir):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    with open(os.path.join(data_dir, 'train_images.pickle'), 'wb') as f:
        pickle.dump(train_images, f)

    with open(os.path.join(data_dir, 'train_labels.pickle'), 'wb') as f:
        pickle.dump(train_labels, f)

    with open(os.path.join(data_dir, 'test_images.pickle'), 'wb') as f:
        pickle.dump(test_images, f)

    with open(os.path.join(data_dir, 'test_labels.pickle'), 'wb') as f:
        pickle.dump(test_labels, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='processing data')
    parser.add_argument('--data_dir', help='path to images and labels.')
    args = parser.parse_args()

    preprocess(data_dir=args.data_dir)
