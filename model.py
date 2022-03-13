import json
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def get_labels():
    # Get labels from json and return them in alphanumeric order
    f = open('data.json', 'r')
    data = json.load(f)
    f.close()
    images, labels = [], []
    images_test, labels_test = [], []
    for elem in data['initial_bundle']:
        if elem['subcategory'] and elem['subcategory']['name'] and elem['subcategory']['name'] == 'Cars':
            images.append(bytes(elem['file'], 'utf-8'))
            labels.append([1, 0, 0])
        elif elem['category'] and elem['category']['name'] and elem['category']['name'] == 'Plants':
            images.append(bytes(elem['file'], 'utf-8'))
            labels.append([0, 0, 1])
        else:
            images.append(bytes(elem['file'], 'utf-8'))
            labels.append([0, 1, 0])
    for elem in data['test_bundle']:
        if elem['subcategory'] and elem['subcategory']['name'] and elem['subcategory']['name'] == 'Cars':
            images_test.append(bytes(elem['file'], 'utf-8'))
            labels_test.append([1, 0, 0])
        elif elem['category'] and elem['category']['name'] and elem['category']['name'] == 'Plants':
            images_test.append(bytes(elem['file'], 'utf-8'))
            labels_test.append([0, 0, 1])
        else:
            images_test.append(bytes(elem['file'], 'utf-8'))
            labels_test.append([0, 1, 0])
    return images, labels, images_test, labels_test


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
                        img_augmentation(images[i]).numpy() / 255
                )
            plt.title(class_names[tf.math.argmax(labels[i])])
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
            lbl = 'label: ' + class_names[tf.math.argmax(labels[i])] + \
                  ' predicted: ' + class_names[tf.math.argmax(model.predict(images)[i])]
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
initial_epochs = 25
fine_tune_epochs = 15
patience = (10, 5)
initial_lr = 0.001
fine_tune_lr = 0.00002
dropout_rate = 0.2
batch_size = 32

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
class_names = ['Cars', 'Others', 'Plants']

# Training initiated if no previous weights found
if not os.path.isdir('model'):
    # Loading the data and labels
    images_train, labels_train, images_test, labels_test = get_labels()
    images_train = tf.data.Dataset.from_tensor_slices(images_train).shuffle(3000, reshuffle_each_iteration=True, seed=0).map(parse_image)
    labels_train = tf.data.Dataset.from_tensor_slices(labels_train).shuffle(3000, reshuffle_each_iteration=True, seed=0)
    images_train = tf.data.Dataset.zip((images_train, labels_train)).batch(batch_size)
    images_test = tf.data.Dataset.from_tensor_slices(images_test).map(parse_image)
    labels_test = tf.data.Dataset.from_tensor_slices(labels_test)
    images_test = tf.data.Dataset.zip((images_test, labels_test)).batch(batch_size)
    # images_train.shuffle(3000, reshuffle_each_iteration=True)
    # plot_dataset()

    # Pass where we're training top layers of our NN from scratch
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.Recall(class_id=0, name='recall_cars'),
                           tf.keras.metrics.Recall(class_id=1, name='recall_others'),
                           tf.keras.metrics.Recall(class_id=2, name='recall_plants')])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00005,
        patience=patience[0],
        restore_best_weights=True,
    )

    # Model training
    history = model.fit(images_train,
                        validation_data=images_test,
                        epochs=initial_epochs,
                        verbose=1,
                        callbacks=[lr_callback, early_stop])

    acc = history.history['recall_plants']
    val_acc = history.history['val_recall_plants']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Fine-tuning pass. We'll train all layers of our model with fine learning rate
    base_model.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.Recall(class_id=0, name='recall_cars'),
                           tf.keras.metrics.Recall(class_id=1, name='recall_others'),
                           tf.keras.metrics.Recall(class_id=2, name='recall_plants')])

    total_epochs = history.epoch[-1] + 1 + fine_tune_epochs
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
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

    acc += history_fine.history['recall_plants']
    val_acc += history_fine.history['val_recall_plants']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    # Evaluating model on a test dataset with weights from early stopping callback
    print()
    print('Evaluating...')
    model.trainable = False
    model.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.Recall()])
    history_eval = model.evaluate(images_test)
    # plot_results()
    # plot_predict()

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
