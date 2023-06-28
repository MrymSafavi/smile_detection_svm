import pickle

import cv2
import numpy as np

from image import Image

# Load trained SVM model
MODEL = pickle.load(open('detect_smile_model.pkl', 'rb'))
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


class UseModelOnClip:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def feature_extracting(self, frame, faces):
        face_gray_img = Image(frame)
        face_gray_img.filter_blur()
        l_img = face_gray_img.lip_detection(faces)

        lip_image = Image(l_img)
        lip_image.make_img_gray()
        hog_feature = lip_image.apply_hog()
        lbp_feature = lip_image.apply_lbp()
        histogram = lip_image.apply_histogram(lbp_feature)

        return hog_feature, histogram

    def start(self):

        while True:
            # Read frame from video
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect faces in frame
            faces = FACE_CASCADE.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


            for (x, y, w, h) in faces:

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                hog_feature, histogram = self.feature_extracting(frame, faces)
                feats = np.concatenate((hog_feature, histogram))

                # Predict smile using SVM model
                smile_pred = MODEL.predict(feats.reshape(1, -1))[0]

                c = None
                if smile_pred == 1:
                    label = 'Smiling'
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    c = (0, 255, 0)
                else:
                    label = 'Not Smiling'
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    c = (0, 0, 255)

                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

            # Display frame with detected faces and smiles
            cv2.imshow('Smile detection', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def end(self):
        self.cap.release()
        self.cv2.destroyAllWindows()


def main():
    use_model = UseModelOnClip()
    use_model.start()
    use_model.end()


if __name__ == '__main__':
    main()
