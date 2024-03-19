import cv2
import numpy as np

import typing
import abc

import pytesseract


class DigitRecognizer(typing.Protocol):
    def preprocess(self, image) -> np.ndarray: ...

    def recognize(self, image) -> int: ...

    def do_process(self, image) -> int: ...


class BaseDigitRecognizer(abc.ABC, DigitRecognizer):
    def preprocess(self, image) -> np.ndarray:
        return image

    def recognize(self, image) -> int:
        preprocessed = self.preprocess(image)
        return self.do_process(preprocessed)

    @abc.abstractmethod
    def do_process(self, image) -> int:
        raise NotImplementedError


class TesseractRecognizer(BaseDigitRecognizer):
    def preprocess(self, image) -> np.ndarray:
        _, bw_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Fill in possible borders with white :(
        bw_image[:8, :] = 255
        bw_image[-8:, :] = 255
        bw_image[:, :8] = 255
        bw_image[:, -8:] = 255

        return bw_image

    def do_process(self, image) -> int:
        """
        BIG BRAIN MOVE
        """
        # If 90% of the image is black - just return -1
        if (image == 0).sum() / image.size > 0.9:
            return -1

        vals = []
        configs = [
            "-l osd --psm 10 --oem 1",
            "--psm 10 --oem 1",
        ]
        for config in configs:
            try:
                output = pytesseract.image_to_string(image, config=config).strip()
                val = int(output[:1])
                return val
            except ValueError:
                vals.append(-1)

        return max(vals)


class KerasRecognizer(BaseDigitRecognizer):
    def __init__(self, model):
        self.model = model

    def preprocess(self, image) -> np.ndarray:
        _, bw_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Fill in possible borders with white :(
        bw_image[:8, :] = 255
        bw_image[-8:, :] = 255
        bw_image[:, :8] = 255
        bw_image[:, -8:] = 255

        if (bw_image == 255).sum() / image.size > 0.99:
            return None

        # cv2.imwrite(f"digits/single-digit-{uuid.uuid4()}.jpg", bw_image)
        image = cv2.resize(bw_image, (28, 28))
        image = image.reshape(1, 28, 28, 1)
        image = image / 255.0
        return image

    def do_process(self, image) -> int:
        if image is None:
            return -1

        prediction = self.model.predict([image], verbose=0)
        return np.argmax(prediction)
