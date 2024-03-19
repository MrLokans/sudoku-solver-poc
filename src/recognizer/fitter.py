import tensorflow as tf


if __name__ == "__main__":
    # Create a dataset of image and labels from the directory
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        "for_fitting",
        image_size=(28, 28),
        color_mode="grayscale",
        batch_size=32,
        label_mode="int",
        shuffle=True,
    )

    # Normalize pixel values to [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))

    model = tf.keras.models.load_model("mnist.keras")
    # Fit the model on the new data
    model.fit(train_dataset, epochs=10)

    model.save("mnist-fit.keras")
