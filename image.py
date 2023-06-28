import cv2
import numpy as np
from mtcnn import MTCNN
from skimage.feature import hog, local_binary_pattern

HOG_PARAMS = {'orientations': 10, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1), 'block_norm': 'L2-Hys',
              'feature_vector': True}

LBP_PARAMS = {'n_points': 8, 'radius': 1, 'method': 'uniform'}


class Image:
    def __init__(self, img):
        self.img = img

    def filter_blur(self):
        median_filter = cv2.medianBlur(self.img, 3)
        self.img = cv2.GaussianBlur(median_filter, (3, 3), 2)

    def resize_image(self):
        self.img = cv2.resize(self.img, (100, 100), interpolation=cv2.INTER_AREA)

    def make_img_gray(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image using a CascadeClassifier object
    def face_detection(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(self.img, scaleFactor=1.1, minNeighbors=5)
        return faces

    def crop_image(self, f_image, x, y, width, height):
        cropped_image = f_image[y:y + height, x:x + width]
        return cropped_image

    def lip_detection(self, faces):
        # Create an MTCNN object for detecting lips
        detector = MTCNN()
        if len(self.face_detection()) == 0:
            return cv2.resize(self.img, (70, 70), interpolation=cv2.INTER_AREA)
        else:
            face_img = self.img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
            results = detector.detect_faces(face_img)
            if results:
                left_lip_x, left_lip_y = results[0]['keypoints']['mouth_left']
                right_lip_x, right_lip_y = results[0]['keypoints']['mouth_right']
                crop = self.crop_image(face_img, left_lip_x - 6, left_lip_y - 7, right_lip_x + 10,
                                       right_lip_y + 10)
                return cv2.resize(crop, (70, 70), interpolation=cv2.INTER_AREA)
            else:
                return cv2.resize(face_img, (70, 70), interpolation=cv2.INTER_AREA)

    def apply_hog(self):
        hog_features = hog(self.img, **HOG_PARAMS)
        return hog_features

    def apply_lbp(self):
        lbp_features = local_binary_pattern(self.img, 16, 2)
        return lbp_features

    def apply_histogram(self, lbp_features):
        n_bins = int(lbp_features.max() + 1)
        hist, _ = np.histogram(lbp_features, bins=n_bins, range=(0, n_bins), density=True)
        return hist
