import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import os

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

total_files = 0
for label in ['covid', 'normal']:
    image_folder = f'dataset/{label}'
    total_files += len(os.listdir(image_folder))

data = []
labels = []
processed_files = 0

for label in ['covid', 'normal']:
    image_folder = f'dataset/{label}'
    for filename in os.listdir(image_folder):
        processed_files += 1
        
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'Analisando imagem: {filename} ({processed_files}/{total_files} - {processed_files/total_files:.2%} concluído)')
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            features = extract_lbp_features(image)
            data.append(features)
            labels.append(1 if label == 'covid' else 0)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=200)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Matriz de Confusão:")
print(conf_matrix)
print(f"Acurácia: {accuracy * 100:.2f}%")
