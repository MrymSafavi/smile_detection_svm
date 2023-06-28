import os

import cv2
import joblib
import numpy as np

from image import Image

model = joblib.load('tmp/detect_smile_model1.pkl')


# all part completely explain in report doc.

class Extractor:

    def __init__(self):

        # matrix that each row is the feature vector of one data
        self.features_extracted = []

    def apply_extracting(self, path):
        data_dir = path
        counter = 0

        for filename in os.listdir(data_dir):
            counter += 1
            print(counter)
            print(filename)
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(data_dir, filename))
                main_image = Image(img)
                main_image.filter_blur()
                main_image.resize_image()

                faces = main_image.face_detection()
                l_img = main_image.lip_detection(faces)

                lip_image = Image(l_img)
                lip_image.make_img_gray()

                hog_feature = lip_image.apply_hog()
                lbp_feature = lip_image.apply_lbp()
                histogram = lip_image.apply_histogram(lbp_feature)

                # hog_features = StandardScaler().fit_transform(hog_features.reshape(1, -1))
                # lbp_features = StandardScaler().fit_transform(lbp_features.reshape(1, -1))

                self.features_extracted.append(np.concatenate((hog_feature, histogram)))
