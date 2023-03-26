import cv2
import numpy as np
from mtcnn import MTCNN
import os
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Параметры обработки видео
video_path = "path/to/video"
video_frame_save_path = "path/to/save/video/frames"
face_cutout_save_path = "path/to/save/face/cutouts"
dataset_save_path = "path/to/save/dataset"

video_folder = '/content/drive/MyDrive/Dyploma/dfdc_train_part_22'
json_file = '/content/drive/MyDrive/Dyploma/dfdc_train_part_22/metadata.json'

with open(json_file, 'r') as f:
    data = json.load(f)
file_list = data['file_list']
labels = data['labels']

# Инициализация MTCNN для детектирования лиц на кадрах
detector = MTCNN()

# Разделение видео на кадры и детектирование лиц
vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 0
while success:
    faces = detector.detect_faces(image)
    for face in faces:
        x, y, w, h = face['box']
        face_image = image[y:y+h, x:x+w]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # Сохранение лиц в отдельную директорию
        cv2.imwrite(os.path.join(face_cutout_save_path, f"face_{count}.jpg"), face_image)
    success, image = vidcap.read()
    count += 1

# Создание датасета из вырезанных лиц и соответствующих меток
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    face_cutout_save_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training')

validation_generator = datagen.flow_from_directory(
    face_cutout_save_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# Загрузка предобученной модели Xception
base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(224,224,3))

# Добавление глобального пулинга и полносвязного слоя
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1, activation='sigmoid')(x)

# Создание модели
model = Model(inputs=base_model.input, outputs=x)

# Замораживание весов базовой модели
for layer in base_model.layers:
    layer.trainable = False

# Компиляция модели
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])