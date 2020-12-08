import argparse
import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

CKPT_1_DIR = "data_1_ckpt"
CKPTS_ALL_DIR = "data_all_ckpts"

CKPT_NAME_PATTERN = "cp-{epoch:04d}.ckpt"
CKPT_NAME = "cp.ckpt"

#example from https://www.tensorflow.org/tutorials/keras/save_and_load

# Define a simple sequential model
def create_model():
      model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
      ])

      model.compile(optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.metrics.SparseCategoricalAccuracy()])

      return model

def get_example_dataset():

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    return train_labels, test_labels, train_images, test_images


def save_all_ckpt(train_images, train_lables, data_dir=CKPT_1_DIR, cp_name=CKPT_NAME, epochs=10):

    os.makedirs(data_dir, exist_ok=True)

    # Create a basic model instance
    model = create_model()

    checkpoint_path = "%s/%s" % (data_dir, cp_name)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Train the model with the new callback
    model.fit(train_images, 
              train_labels,  
              epochs=epochs,
              validation_data=(test_images, test_labels),
              callbacks=[cp_callback])  # Pass callback to training

    # This may generate warnings related to saving the state of the optimizer.
    # These warnings (and similar warnings throughout this notebook)
    # are in place to discourage outdated usage, and can be ignored.

    return model

def eval_from_ckpt(test_images, test_labels, ckpt_path):

    # Create a basic model instance
    model = create_model()

    # Evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    # Loads the weights
    print("Trying to load weights from %s" % ckpt_path)
    model.load_weights(ckpt_path) #checkpoint_path: data_dir/cp-0050.ckpt or data_dir/cp.ckpt

    # Re-evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


def save_every_n_ckpt(train_images, train_lables,
                      data_dir=CKPTS_ALL_DIR, cp_name_pattern=CKPT_NAME_PATTERN,
                      every_epochs=5, epochs=50):

    os.makedirs(data_dir, exist_ok=True)

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "%s/%s" % (data_dir, cp_name_pattern)

    batch_size = 32

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=every_epochs*batch_size)

    # Create a new model instance
    model = create_model()

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # Train the model with the new callback
    model.fit(train_images, 
              train_labels,
              epochs=epochs,
              callbacks=[cp_callback],
              validation_data=(test_images, test_labels),
              verbose=0)

    return model


def slp(data_dir):
    #TODO: data_dir is fake here

    train_labels, test_labels, train_images, test_images = get_example_dataset()

    # save checkpoint(s)
    model_1_ckpt = save_all_ckpt(train_images, train_lables)
    model_ckpts = save_every_n_ckpt(train_images, train_lables)

    # save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_1_ckpt.save( models_dir, "model_1_ckpt")
    model_ckpts.save( models_dir, "model_ckpts")

    # eval test images with lables from checkpoints
    eval_from_ckpt(test_images, test_labels, CKPT_NAME)
    eval_from_ckpt(test_images, test_labels, CKPT_NAME_PATTERN)

    # or tf.keras.models.load_model('saved_model/my_model')
    model_1_ckpt.summary()
    model_ckpts.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='processing data')
    parser.add_argument('--data_dir', help='path to images and labels.')
    args = parser.parse_args()

    slp(args.data_dir)
