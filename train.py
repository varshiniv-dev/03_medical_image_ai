import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

IMG = 128
BATCH = 32

np.random.seed(42)
tf.random.set_seed(42)

# Real dataset mode:
# data/normal and data/abnormal
root = "data"

has_real = (
    os.path.isdir(os.path.join(root, "normal"))
    and os.path.isdir(os.path.join(root, "abnormal"))
)

if has_real:

    train_ds = tf.keras.utils.image_dataset_from_directory(
        root,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG, IMG),
        batch_size=BATCH,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        root,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG, IMG),
        batch_size=BATCH,
        label_mode="binary"
    )

    class_names = train_ds.class_names

else:

    # Synthetic fallback dataset.
    # Normal images contain background noise.
    # Abnormal images contain a bright circular region.
    #
    # IMPORTANT:
    # Images are generated in the 0-255 range because
    # the model contains a Rescaling(1./255) layer.

    def make(n, abnormal):

        x = np.random.normal(
            45,
            15,
            (n, IMG, IMG, 3)
        ).clip(0, 255)

        if abnormal:

            for i in range(n):

                cx, cy = np.random.randint(30, 98, 2)
                r = np.random.randint(10, 28)

                yy, xx = np.ogrid[:IMG, :IMG]

                mask = (
                    (xx - cx) ** 2
                    + (yy - cy) ** 2
                    < r ** 2
                )

                x[i, mask, :] = np.random.uniform(
                    165,
                    240,
                    size=(mask.sum(), 3)
                )

        else:

            for i in range(n):

                x[i] += np.random.normal(
                    0,
                    5,
                    (IMG, IMG, 1)
                )

        return x.clip(0, 255).astype("float32")

    n = 1600

    x0 = make(n // 2, False)
    x1 = make(n // 2, True)

    x = np.concatenate([x0, x1])

    y = np.concatenate([
        np.zeros(len(x0)),
        np.ones(len(x1))
    ])

    idx = np.random.permutation(len(x))

    x = x[idx]
    y = y[idx]

    split = int(0.8 * len(x))

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x[:split], y[:split])
    ).batch(BATCH)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (x[split:], y[split:])
    ).batch(BATCH)

    class_names = ["normal", "abnormal"]


# CNN model

model = keras.Sequential([

    layers.Input((IMG, IMG, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(
        32,
        3,
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        3,
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        3,
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)


os.makedirs("artifacts", exist_ok=True)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)


model.save(
    "artifacts/medical_classifier.keras"
)


print("Classes:", class_names)

print(
    "Saved artifacts/medical_classifier.keras"
)


# Training accuracy plot

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="train"
)

plt.plot(
    history.history["val_accuracy"],
    label="validation"
)

plt.title("CNN Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    "artifacts/training_accuracy.png",
    dpi=160
)

plt.close()
