import pandas as pd
import os
import numpy as np
from PIL import Image
from natsort import natsorted
from google.colab import drive
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

drive.mount('/content/drive')

!unzip "/content/drive/MyDrive/Final_project_Data/train_data_unlabeled.zip" -d "./train_data"
!unzip "/content/drive/MyDrive/Final_project_Data/test_data_unlabeled.zip" -d "./test_data"

def images_to_array_sorted(path):
    images = []
    image_filenames = natsorted(os.listdir(path))
    sum_channels = np.zeros(3)
    sum_squares_channels = np.zeros(3)
    pixel_count = 0

    for image in image_filenames:
        image_path = os.path.join(path,image)
        img = Image.open(image_path)
        img_array = np.array(img, dtype=np.float32)
        img_array /= 255.0
        rgb_image = img_array[:, :, :3]
        images.append(rgb_image)
    images = np.array(images)
    return images

train_path = './train_data/train_data_unlabeled'
train_images = images_to_array_sorted(train_path)

test_path = './test_data/test_data_unlabeled'
test_images = images_to_array_sorted(test_path)

y_train_csv = '/content/drive/MyDrive/Final_project_Data/y_train.csv'
y_train_df = pd.read_csv(y_train_csv)
labels = y_train_df['target'].values


plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i])
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(45, 51, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(4, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience=2,
    verbose=1,
    mode='max',
    restore_best_weights=True
)


history = model.fit(train_images, labels, epochs=10, batch_size=16, validation_split=0.2, verbose=2, callbacks=[early_stopping])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)


ids = np.arange(0, len(predicted_classes))
df = pd.DataFrame({
    'id': ids,
    'target': predicted_classes
})
csv_file_path = '/content/my_submission_diff.csv'
df.to_csv(csv_file_path, index=False)


plt.figure(figsize=(10, 10))
for i in range(60):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i])
    plt.title(f"Predicted: {predicted_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()