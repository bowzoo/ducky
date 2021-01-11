import argparse
import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)


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

def slp(data_dir):
    #TODO: data_dir is fake
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


    os.makedirs("training_1", exist_ok=True)
    os.makedirs("training_2", exist_ok=True)
    os.makedirs("saved_model_pip", exist_ok=True)

    # Create a basic model instance
    model = create_model()

    # Display the model's architecture
    model.summary()




    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Train the model with the new callback
    model.fit(train_images, 
              train_labels,  
              epochs=10,
              validation_data=(test_images, test_labels),
              callbacks=[cp_callback])  # Pass callback to training

    # This may generate warnings related to saving the state of the optimizer.
    # These warnings (and similar warnings throughout this notebook)
    # are in place to discourage outdated usage, and can be ignored.




    # Create a basic model instance
    model = create_model()

    # Evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))



    # Loads the weights
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))






    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    batch_size = 32

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=5*batch_size)

    # Create a new model instance
    model = create_model()

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # Train the model with the new callback
    model.fit(train_images, 
              train_labels,
              epochs=50, 
              callbacks=[cp_callback],
              validation_data=(test_images, test_labels),
              verbose=0)




    latest = tf.train.latest_checkpoint(checkpoint_dir)



    # Create a new model instance
    model = create_model()

    # Load the previously saved weights
    model.load_weights(latest)

    # Re-evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("zRestored model, accuracy: {:5.2f}%".format(100 * acc))



    # Create and train a new model instance.
    model = create_model()
    model.fit(train_images, train_labels, epochs=5)



    new_model = tf.keras.models.load_model('saved_model_pip/my_model')

    # Check its architecture
    new_model.summary()


    model.save('saved_model_pip/my_model') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='processing data')
    parser.add_argument('--data_dir', help='path to images and labels.')
    args = parser.parse_args()

    slp(args.data_dir)
