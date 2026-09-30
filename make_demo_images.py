import numpy as np
from PIL import Image
import os

IMG = 128
np.random.seed(123)

os.makedirs("demo_images", exist_ok=True)


def make_image(abnormal=False):
    x = np.random.normal(
        45,
        15,
        (IMG, IMG, 3)
    ).clip(0, 255)

    if abnormal:
        cx, cy = 64, 64
        r = 25

        yy, xx = np.ogrid[:IMG, :IMG]

        mask = (
            (xx - cx) ** 2 +
            (yy - cy) ** 2
            < r ** 2
        )

        x[mask, :] = 220

    return x.astype("uint8")


normal = make_image(False)
abnormal = make_image(True)

Image.fromarray(normal).save(
    "demo_images/demo_normal.png"
)

Image.fromarray(abnormal).save(
    "demo_images/demo_abnormal.png"
)

print("Created:")
print("demo_images/demo_normal.png")
print("demo_images/demo_abnormal.png")
