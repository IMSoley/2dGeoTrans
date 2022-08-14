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

## Results

## Relevant paper

[M. B. Islam, W. Lai-Kuan, W. Chee-Onn and K. -L. Low, "Stereoscopic image warping for enhancing composition aesthetics," 2015 3rd IAPR Asian Conference on Pattern Recognition (ACPR), 2015, pp. 645-649, doi: 10.1109/ACPR.2015.7486582.](https://ieeexplore.ieee.org/document/7486582)
