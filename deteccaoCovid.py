import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import os

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def get_total_files(dataset_path, categories):
    total_files = 0
    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        total_files += len(os.listdir(folder_path))
    return total_files

def load_dataset(dataset_path, categories):
    data = []
    labels = []
    total_files = get_total_files(dataset_path, categories)
    processed_files = 0

    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        label = 1 if category == 'covid' else 0
        for filename in os.listdir(folder_path):
            processed_files += 1
            clear_terminal()
            print(f'Processing image: {filename} ({processed_files}/{total_files} - {processed_files/total_files:.2%} completed)')

            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                features = extract_lbp_features(image)
                data.append(features)
                labels.append(label)

    return np.array(data), np.array(labels)

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    dataset_path = 'dataset'
    categories = ['covid', 'normal']

    data, labels = load_dataset(dataset_path, categories)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=200)

    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()