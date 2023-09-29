from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from dataset import Dataset

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pickle

dataset_directory = "Dataset"
compressed_dataset = "dataset.npz"
target_size = (160, 160)

encoded_X = []
Y = []

dataset = Dataset(target_size)

try:
    X, encoded_X, Y = dataset.load(compressed_dataset, method="from_numpy")
    print(f"[INFO] Found '{compressed_dataset}'. Loaded compressed dataset successfully.")
except FileNotFoundError:
    print(f"[INFO] Not found '{compressed_dataset}'. Loading new dataset from directory instead.")
    X, encoded_X, Y = dataset.load(dataset_directory, method="from_directory")
    np.savez_compressed(compressed_dataset, X=X, encoded_X=encoded_X, Y=Y)
    print(f"[INFO] Saved '{compressed_dataset}' successfully.")

dataset.show()

label_encoder = LabelEncoder()
label_encoder.fit(Y)
Y = label_encoder.transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(encoded_X, Y, test_size=0.2, random_state=42)
model = svm.SVC(kernel="linear", C=0.8, gamma="scale", probability=True)
model.fit(X_train, Y_train)

Y_train_predict = model.predict(X_train)
Y_test_predict = model.predict(X_test)
train_accuracy = accuracy_score(Y_train, Y_train_predict)
test_accuracy = accuracy_score(Y_test, Y_test_predict)
confusion_matrix = confusion_matrix(Y_test, Y_test_predict)

print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

inverse_transform_Y = np.unique(label_encoder.inverse_transform(Y))

plt.figure(figsize=(12, 8))
plt.title("Confusion matrix")
sns.heatmap(confusion_matrix, xticklabels=inverse_transform_Y,
            yticklabels=inverse_transform_Y, annot=True, fmt=".0f", cmap=plt.cm.Blues)

plt.show()

pickle.dump(model, open("model.svm", "wb"))
print("[INFO] Saved model successfully.")
