import cv2
import abc
from utils.boundingbox import Box
from typing import List, Mapping, Union
from collections import OrderedDict
from numpy import ndarray as CVImage
from pathlib import Path

# type annotation aliases
FileName = str


def get_the_haarcascade(filename: str) -> Path:
    cascade_dir = (Path(cv2.__file__) / '../../../../share/OpenCV/haarcascades/').resolve()
    # cascade_dir = ('../cvdata/')
    if not cascade_dir.exists():
        raise RuntimeError('Cannot find OpenCV haarcascades')
    file = cascade_dir / filename
    if not file.exists():
        raise RuntimeError('Cannot find file {file}'.format(file=file))
    return file


class Feature(Box):
    def __init__(self, weight: float, label: str, *args, **kwargs) -> None:
        self.weight = weight
        self.label = label
        super().__init__(*args, **kwargs)

    def __lt__(self, other):
        # self < other
        # this enables sorting
        return self.weight < other.weight

    def __mul__(self, factor: float) -> 'Feature':
        box = super().__mul__(factor)
        return self.__class__(
            label=self.label,
            weight=self.weight * factor,
            **box.__dict__, )

    def serialize(self, precision: int=3) \
            -> Mapping[str, Union[str, float]]:

        def floatformat(f):
            return round(f, precision)

        return OrderedDict([
            ('label', self.label),
            ('x', floatformat(self.left)),
            ('y', floatformat(self.top)),
            ('width', floatformat(self.width)),
            ('height', floatformat(self.height)),
            ('weight', floatformat(self.weight)),
        ])

    @classmethod
    def deserialize(cls, data: dict) -> 'Feature':
        left = float(data.get('x'))
        top = float(data.get('y'))
        bottom = top + float(data.get('height'))
        right = left + float(data.get('width'))
        return cls(
            label=data.get('label', 'feature'),
            weight=data.get('weight', 0),
            left=left,
            top=top,
            bottom=bottom,
            right=right)


class FeatureDetector(abc.ABC):
    @abc.abstractmethod
    def __init__(self, n: int) -> None:
        ...

    @abc.abstractmethod
    def detect_features(self, fn: FileName) -> List[Feature]:
        ...

    @staticmethod
    def _opencv_image(fn: str, resize: int = 0) -> CVImage:
        cv_image = cv2.imread(fn)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        if resize > 0:
            w, h = cv_image.shape[1::-1]  # type: int, int
            multiplier = (resize**2 / (w * h))**0.5
            dimensions = tuple(int(round(d * multiplier)) for d in (w, h))
            cv_image = cv2.resize(
                cv_image, dimensions, interpolation=cv2.INTER_AREA)
        return cv_image

    @staticmethod
    def _resize_feature(feature: Feature, cv_image: CVImage) -> Feature:
        img_h, img_w = cv_image.shape[:2]  # type: int, int
        feature = Feature(
            label=feature.label,
            weight=feature.weight / (img_w * img_h),
            left=max(0, feature.left / img_w),
            top=max(0, feature.top / img_h),
            right=min(1, feature.right / img_w),
            bottom=min(1, feature.bottom / img_h), )
        return feature


class MFeatureDetector(FeatureDetector):
    def __init__(self, n: int = 3, imagesize: int = 200) -> None:
        self._number = n
        self._size = imagesize
        self._circles = [m / n for m in range(1, n + 1)]

    def detect_features(self, fn: FileName) -> List[Feature]:
        cv_image = self._opencv_image(fn, self._size)
        img_h, img_w = cv_image.shape[:2]
        middle = Feature(0, 'mock keypoint', 0, 0, img_w, img_h)
        middle.width = middle.height = min(img_w, img_h)
        middle = self._resize_feature(middle, cv_image)
        return [middle * size for size in self._circles]


class KeypointDetector(FeatureDetector):
    LABEL = 'ORB keypoint'

    def __init__(self,
                 n: int = 10,
                 padding: float = 1.0,
                 imagesize: int = 200,
                 **kwargs) -> None:
        self._imagesize = imagesize
        self._padding = padding
        _kwargs = {
            "nfeatures": n + 1,
            "scaleFactor": 1.5,
            "patchSize": self._imagesize // 10,
            "edgeThreshold": self._imagesize // 10,
            "scoreType": cv2.ORB_FAST_SCORE,
        }
        _kwargs.update(kwargs)
        self._detector = cv2.ORB_create(**_kwargs)

    def detect_features(self, fn: str) -> List[Feature]:
        cv_image = self._opencv_image(fn, self._imagesize)
        keypoints = self._detector.detect(cv_image)
        features = [self._keypoint_to_feature(kp) for kp in keypoints]
        features = [self._resize_feature(ft, cv_image) for ft in features]
        return sorted(features, reverse=True)

    def _keypoint_to_feature(self, kp: cv2.KeyPoint) -> Feature:
        x, y = kp.pt
        radius = kp.size / 2
        weight = radius * kp.response**2
        return Feature(
            label=self.LABEL,
            weight=weight,
            left=x - radius,
            top=y - radius,
            right=x + radius,
            bottom=y + radius) * self._padding


class Cascade:
    def __init__(self,
                 label: str,
                 fn: FileName,
                 size: float = 1,
                 weight: float = 100) -> None:
        self.label = label
        self.size = size
        self.weight = weight
        self._file = str(get_the_haarcascade(fn))
        self.classifier = cv2.CascadeClassifier(self._file)
        if self.classifier.empty():
            msg = ('The input file: "{}" is not a valid '
                   'cascade classifier').format(self._file)
            raise RuntimeError(msg)


class FaceDetector(FeatureDetector):
    _CASCADES = [
        Cascade(
            'frontal face',
            'haarcascade_frontalface_default.xml',
            size=1.0,
            weight=100),
        Cascade(
            'alt face',
            'haarcascade_frontalface_alt.xml',
            size=1.2,
            weight=100),
        Cascade(
            'profile face', 'haarcascade_profileface.xml', size=0.9,
            weight=50),
    ]

    def __init__(self,
                 n: int = 10,
                 padding: float = 1.2,
                 imagesize: int = 600,
                 **kwargs) -> None:
        self._number = n
        self._imagesize = imagesize
        self._padding = padding
        self._cascades = self._CASCADES
        minsize = max(25, imagesize // 20)
        self._kwargs = {
            "minSize": (minsize, minsize),
            "scaleFactor": 1.2,
            "minNeighbors": 5,
        }
        self._kwargs.update(kwargs)

    def detect_features(self, fn: FileName) -> List[Feature]:
        cv_image = self._opencv_image(fn, self._imagesize)
        features = []  # type: List[Feature]

        for cascade in self._cascades:
            padding = self._padding * cascade.size
            detect = cascade.classifier.detectMultiScale
            faces = detect(cv_image, **self._kwargs)

            for left, top, width, height in faces:
                weight = height * width * cascade.weight
                face = Feature(
                    label=cascade.label,
                    weight=weight,
                    left=left,
                    top=top,
                    right=left + width,
                    bottom=top + height, )
                face = face * padding
                face = self._resize_feature(face, cv_image)
                features.append(face)

        return sorted(features, reverse=True)[:self._number]


class HybridDetector(FeatureDetector):
    BREAKPOINT = 0.15

    def __init__(self, n=10) -> None:
        self._number = n
        self.primary = FaceDetector(n)
        self.fallback = KeypointDetector(n)
        self.breakpoint = self.BREAKPOINT

    def detect_features(self, fn: FileName) -> List[Feature]:
        faces = self.primary.detect_features(fn)
        if faces and sum(faces).size > self.breakpoint:
            return faces
        features = faces + self.fallback.detect_features(fn)
        return features[:self._number]
