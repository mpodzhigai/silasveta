import json
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil


def dataset_by_directory():

    # Preparing data for keras dataset util

    os.mkdir(os.path.join('data', 'test'))
    os.mkdir(os.path.join('data', 'test', 'Cars'))
    os.mkdir(os.path.join('data', 'test', 'Plants'))
    os.mkdir(os.path.join('data', 'test', 'Others'))
    os.mkdir(os.path.join('data', 'train'))
    os.mkdir(os.path.join('data', 'train', 'Cars'))
    os.mkdir(os.path.join('data', 'train', 'Plants'))
    os.mkdir(os.path.join('data', 'train', 'Others'))
    f = open('data.json', 'r')
    data = json.load(f)
    f.close()
    for elem in data['initial_bundle']:
        if elem['subcategory'] and elem['subcategory']['name'] and elem['subcategory']['name'] == 'Cars':
            shutil.copyfile(elem['file'], os.path.join('data', 'train', 'Cars', elem['file'].split(os.sep)[-1]))
        elif elem['category'] and elem['category']['name'] and elem['category']['name'] == 'Plants':
            shutil.copyfile(elem['file'], os.path.join('data', 'train', 'Plants', elem['file'].split(os.sep)[-1]))
        else:
            shutil.copyfile(elem['file'], os.path.join('data', 'train', 'Others', elem['file'].split(os.sep)[-1]))
    for elem in data['test_bundle']:
        if elem['subcategory'] and elem['subcategory']['name'] and elem['subcategory']['name'] == 'Cars':
            shutil.copyfile(elem['file'], os.path.join('data', 'test', 'Cars', elem['file'].split(os.sep)[-1]))
        elif elem['category'] and elem['category']['name'] and elem['category']['name'] == 'Plants':
            shutil.copyfile(elem['file'], os.path.join('data', 'test', 'Plants', elem['file'].split(os.sep)[-1]))
        else:
            shutil.copyfile(elem['file'], os.path.join('data', 'test', 'Others', elem['file'].split(os.sep)[-1]))
    return


def lr_scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.90


def plot_dataset():
    plt.figure(figsize=(10, 10))
    for images, labels in images_train.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            with tf.device('/cpu:0'):
                plt.imshow(
                        img_augmentation(images[i].numpy().astype("uint8"))
                )
            plt.title(images_train.class_names[labels[i]])
            plt.axis("off")
    plt.show()


def plot_results():
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training accuracy')
    plt.plot(val_acc, label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.show()


def plot_predict():
    plt.figure(figsize=(10, 10))
    for images, labels in images_test.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            lbl = 'label: ' + images_test.class_names[labels[i]] + \
                  ' predicted: ' + images_test.class_names[tf.math.argmax(model.predict(images)[i])]
            plt.title(lbl)
            plt.axis("off")
    plt.show()


def parse_folder(path):
    files = []
    for file in os.listdir(path):
        if file.split('.')[-1].lower() == 'png' or file.split('.')[-1].lower() == 'jpg' or file.split('.')[-1].lower() == 'jpeg':
            files.append(os.path.join(path, file))
    return files


def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.resize_with_pad(image, img_width, img_width)
    return image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Neural network's hyperparameters
img_width = 224
initial_epochs = 20
fine_tune_epochs = 10
patience = (10, 5)
initial_lr = 0.001
fine_tune_lr = 0.00002
dropout_rate = 0.2

# We'll use transfer learning approach with EfficientNetB0 pretrained on ImageNet
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False
inputs = tf.keras.layers.Input(shape=(img_width, img_width, 3), name='input_layer')

# Augmentation to create more training data and prevent overfitting
img_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=0.1),
        tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomZoom(height_factor=(-0.4, 0.0))
    ],
    name='img_augmentation',
)
with tf.device('/cpu:0'):
    x = img_augmentation(inputs)

# Adding custom top layers
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Training initiated if no previous weights found
if not os.path.isdir('model'):
    # Processing and loading the raw dataset
    if not os.path.isdir(os.path.join('data', 'train')):
        dataset_by_directory()
    images_train = tf.keras.utils.image_dataset_from_directory(os.path.join('data', 'train'),
                                                               image_size=(img_width, img_width))
    images_test = tf.keras.utils.image_dataset_from_directory(os.path.join('data', 'test'),
                                                              image_size=(img_width, img_width))
    # plot_dataset()

    # Pass where we're training top layers of our NN from scratch
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['sparse_categorical_accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        min_delta=0.00005,
        patience=patience[0],
        restore_best_weights=True,
    )

    # Model training
    history = model.fit(images_train,
                        epochs=initial_epochs,
                        validation_data=images_test,
                        verbose=1,
                        callbacks=[lr_callback, early_stop])

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Fine-tuning pass. We'll train all layers of our model with fine learning rate
    base_model.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['sparse_categorical_accuracy'])

    total_epochs = history.epoch[-1] + 1 + fine_tune_epochs
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        min_delta=0.00005,
        patience=patience[1],
        restore_best_weights=True,
    )
    history_fine = model.fit(images_train,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1] + 1,
                             validation_data=images_test,
                             verbose=1,
                             callbacks=[early_stop])

    model.save_weights(os.path.join('model', 'chkp'))

    acc += history_fine.history['sparse_categorical_accuracy']
    val_acc += history_fine.history['val_sparse_categorical_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    # Evaluating model on a test dataset with weights from early stopping callback
    print()
    print('Evaluating...')
    model.trainable = False
    model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    history_eval = model.evaluate(images_test)
    # plot_results()
    # plot_predict()

    # Removing processed dataset
    shutil.rmtree(os.path.join('data', 'test'))
    shutil.rmtree(os.path.join('data', 'train'))

# Load weights
model.load_weights(os.path.join('model', 'chkp')).expect_partial()
model.trainable = False

# Run through dataset and predict labels. Cars and plants are printed alongside with confidence in result
for file in parse_folder('data'):
    prediction = model(tf.expand_dims(parse_image(file), axis=0), training=False)[0]
    if tf.math.argmax(prediction) == 0:
        print('File', file.split(os.sep)[-1], 'is a car with', '{:.1%}'.format(prediction[0]), 'confidence')
    elif tf.math.argmax(prediction) == 2:
        print('File', file.split(os.sep)[-1], 'is a plant with', '{:.1%}'.format(prediction[2]), 'confidence')
