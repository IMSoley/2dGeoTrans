# 2dGeoTrans

**PhD Task**: Geometrically transform the stereoscopic 2D image _(object position)_ to a new position based on the photographic composition rules.

- For this task, I used the `rule of thirds` to transform the image to a new position.

- [The rule of thirds](https://www.adobe.com/creativecloud/photography/discover/rule-of-thirds.html) is a composition guideline that places your subject in the left or right third of an image, leaving the other two thirds more open. While there are other forms of composition, the rule of thirds generally leads to compelling and well-composed shots.

**The task was completed on October 13, 2019. Video demo: [2dGeoTrans_poc](https://youtu.be/OlTzrvfv63Y)**

## How to run locally

- Install jupyter notebook: `pip install jupyter notebook`
- Clone the repository: `git clone https://github.com/IMSoley/2dGeoTrans`
- Install dependencies: `pip install -r requirements.txt`
- Open command prompt and type the following:

    ```cmd
    C:\Users\YourName> python
    ```

    ```python
    >>> import cv2
    >>> from pathlib import Path
    >>> (Path(cv2.__file__) / '../../../../share/OpenCV/haarcascades/').resolve()
    # If necessary, create the directory as shown here.
    >>> exit()
    ```

- Copy [haarcascade](cvdata/) files to the above path
- Run the notebook: `jupyter notebook` in the 2dGeoTrans directory

## Technology used

- [OpenCV](https://opencv.org/) - Computer Vision Library
- [Python](https://www.python.org/) - Programming Language
- [Jupyter Notebook](https://jupyter.org/) - Notebook Environment
- [PyTest](https://docs.pytest.org/en/latest/) - Unit Testing Framework
- [Javascript](https://www.javascript.com/) - For formatting the output

## Workflow and results

- Detecting the keypoints and descriptors of the object in the image with ORB detector. ORB is short for [Oriented FAST and Rotated BRIEF](https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html).

    ```python
    # creating the numpy image array
    image_array = cv2.imread(input_image)
    # creating the orb detector
    orb_detector = cv2.ORB_create(3)
    # detecting keypoints and descriptors
    keypoints = orb_detector.detect(image_array)
    ```

<p align="center">
    <img src="https://user-images.githubusercontent.com/13655344/184545605-5267991f-2c69-4d20-a469-a8ec596f6f4b.png">
</p>

- Face detection with Viola-Jones

    ```python
    image_array = cv2.imread(panda_image)
    cascade_file = '' # Path to the cascade file
    viola_jones_classifier = cv2.CascadeClassifier(cascade_file)
    viola_jones_classifier.detectMultiScale(image_array)
    ```

- Face detection with Haar Cascade

    ```python
    feature_detect_and_show('img/panda1.jpg', g_face_detector)
    ```

<p align="center">
    <img src="https://user-images.githubusercontent.com/13655344/184550198-30e8ce52-e5fe-4c77-bbd3-712c300f4e2b.png">
</p>

- Combined feature detector: this detector combines the power of both algorithms `FaceDetector` and `KeypointDetector` through the photographic composition rules.

    ```python
    class HybridDetector(FeatureDetector):

    BREAKPOINT = 0.15

    def __init__(self, n=10) -> None:
        self.primary = FaceDetector(n, padding=1.5)
        self.fallback = KeypointDetector(n, padding=1.2)
        self.breakpoint = self.BREAKPOINT
        self._number = n

    def detect_features(self, fn: FileName) -> List[Feature]:
        faces = self.primary.detect_features(fn)
        if faces and sum(faces).size > self.breakpoint:
            return faces
        features = faces + self.fallback.detect_features(fn)
        return features[:self._number]
    ```

    ```python
    feature_detect_and_show('img/panda1.jpg', HybridDetector())
    ```

<p align="center">
    <img src="https://user-images.githubusercontent.com/13655344/184550461-555062dc-83ca-4dd6-9334-0e0fb4a96c37.png">
</p>

### Final results using the `HybridDetector`

```python
images = [ 
    'img/panda1.jpg',
    'img/panda2.jpg',
    'img/panda3.jpg'
]
detector = HybridDetector()
feature_detect_and_show(images, HybridDetector(), preview=True)
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/13655344/184550815-7afd7537-3d0f-42b1-94fb-44e893feba54.png">
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/13655344/184550836-c60fd50b-6830-40b2-bc46-3b4914741eb8.png">
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/13655344/184550857-b89f8983-0e71-43c9-9837-6e33d492b18e.png">
</p>

**For better results, it is recommended to use human images as the haarcascades are very accurate for human faces.**

## Relevant paper

[M. B. Islam, W. Lai-Kuan, W. Chee-Onn and K. -L. Low, "Stereoscopic image warping for enhancing composition aesthetics," 2015 3rd IAPR Asian Conference on Pattern Recognition (ACPR), 2015, pp. 645-649, doi: 10.1109/ACPR.2015.7486582.](https://ieeexplore.ieee.org/document/7486582)
