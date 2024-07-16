import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Parâmetros globais
img_width, img_height = 64, 64
batch_size = 32
epochs = 20
train_dir = './training'  # Substitua pelo caminho correto
test_dir = './test'    # Substitua pelo caminho correto

def create_data_generators(train_dir, test_dir, img_width, img_height, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator

def create_model(img_width, img_height, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator, epochs, batch_size):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs
    )
    model.save('modelo_gestos.h5')
    return history

def evaluate_model(model, validation_generator, batch_size):
    loss, accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

# Main script
if __name__ == "__main__":
    train_generator, validation_generator = create_data_generators(train_dir, test_dir, img_width, img_height, batch_size)
    num_classes = len(train_generator.class_indices)
    model = create_model(img_width, img_height, num_classes)
    
    train_model(model, train_generator, validation_generator, epochs, batch_size)
    evaluate_model(model, validation_generator, batch_size)
