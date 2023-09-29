import matplotlib.pyplot as plt
import numpy as np

import face_recognition
import os
import cv2


class Dataset:

    def __init__(self, target_size):
        self.target_size = target_size
        self.X = []
        self.encoded_X = []
        self.Y = []

    def load_from_numpy(self, compressed_dataset):
        dataset = np.load(compressed_dataset)
        self.X = dataset["X"]
        self.encoded_X = dataset["encoded_X"]
        self.Y = dataset["Y"]

    def load_from_directory(self, dataset_directory):
        for name in os.listdir(dataset_directory):
            if name == ".DS_Store":
                continue

            print(f"[INFO] Loading person: {name}...")

            process = 1

            image_file = os.listdir(dataset_directory + "/" + name)

            n = len(image_file)

            for each in image_file:
                if not each.endswith("jpg") or each == ".DS_Store":
                    continue

                image = cv2.imread(dataset_directory + "/" + name + "/" + each)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                detected_face = face_recognition.face_locations(gray_image, model="cnn")

                if len(detected_face) == 1:
                    encoded_face = face_recognition.face_encodings(rgb_image, detected_face)[0]
                    top, right, bottom, left = detected_face[0]
                    face = rgb_image[top:bottom, left:right]
                    face = cv2.resize(face, self.target_size)
                    self.X.append(face)
                    self.encoded_X.append(encoded_face)
                    self.Y.append(name)
                else:
                    print(f"[WARNING] {each} was skipped because it either can't detect a face or "
                          f"contains more than one face.")

                print(f"Processing {name}: {process}/{n}")

                process += 1

            print(f"[INFO] Loaded {name} successfully.")

    def load(self, source, method="from_numpy"):
        if method == "from_numpy":
            self.load_from_numpy(source)
        elif method == "from_directory":
            self.load_from_directory(source)
        else:
            print("[ERROR] Loading method must be specified.")
            return

        return np.asarray(self.X), np.asarray(self.encoded_X), np.asarray(self.Y)

    def show(self):
        figure, axis = plt.subplots(nrows=5, ncols=5, figsize=(18, 16))

        random_image_index = np.random.choice(len(self.X), 30)

        for r in range(5):
            for c in range(5):
                axis[r][c].imshow(self.X[random_image_index[5 * r + c]])
                axis[r][c].axis("off")

        plt.show()

