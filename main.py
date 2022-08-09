import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix

from prepareData import VOCAB_SIZE, WIN_SIZE, x_train, y_train, x_test, y_test, x_val, y_val

CLASS_LIST = ['neutral', 'dump', 'pump']

def make_mod(
    VOCAB_SIZE,
    WIN_SIZE,
    CLASS_COUNT
):

    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, 32, input_length=WIN_SIZE))
    model.add(SpatialDropout1D(0.2))
    model.add(BatchNormalization())
    model.add(Conv1D(2, 1, activation='relu'))
    model.add(Conv1D(2, 1, activation='relu'))
    model.add(MaxPooling1D(1))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(CLASS_COUNT, activation='softmax'))
    return model

def compile_train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    optimizer='adam',
    epochs=50,
    batch_size=128,
    figsize=(20, 5)
):
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    plot_model(model, dpi=60, show_shapes=True)

    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('model learning plot')
    ax1.plot(history.history['accuracy'],
               label='train accuracy')
    ax1.plot(history.history['val_accuracy'],
               label='test accuracy')
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('share of correct answers')
    ax1.legend()

    ax2.plot(history.history['loss'],
               label='train mistakes')
    ax2.plot(history.history['val_loss'],
               label='validate mistakes')
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mistakes')
    ax2.legend()
    plt.show()

    return model

def eval_model(
    model,
    x,
    y_true,  # text sample of text marks
    class_labels,
    cm_round=2,  # round parameter
    title='',
    figsize=(25, 25)
):

    y_pred = model.predict(x)

    # mistakes matrix
    cm = confusion_matrix(np.argmax(y_true, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')

    # matrix round
    cm = np.around(cm, cm_round)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Neural network {title}: normalized mistakes matrix', fontsize=18)
    plt.xlabel('prediction classes', fontsize=26)
    plt.ylabel('actual classes', fontsize=26)
    fig.autofmt_xdate(rotation=45)
    plt.show()

    print('-' * 100)
    print(f'neuralnetwork: {title}')

    for cls in range(len(class_labels)):
        # max confidence
        cls_pred = np.argmax(cm[cls])
        msg = 'true :-)' if cls_pred == cls else 'false :-('
        print('Class: {:<20} {:3.0f}% classify as {:<20} - {}'.format(
            class_labels[cls],
            100. * cm[cls, cls_pred],
            class_labels[cls_pred],
            msg
        ))

    print('\naverage accuracy: {:3.0f}%'.format(100. * cm.diagonal().mean()))

def compile_train_eval_model(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    class_labels=CLASS_LIST,
    title='',
    optimizer='adam',
    epochs=50,
    batch_size=128,
    graph_size=(20, 5),
    cm_size=(15, 15)
):

    model = compile_train_model(model,
                                x_train, y_train,
                                x_test, y_test,
                                optimizer=optimizer,
                                epochs=epochs,
                                batch_size=batch_size,
                                figsize=graph_size)


    eval_model(model, x_test, y_test,
               class_labels=class_labels,
               title=title,
               figsize=cm_size)


    return model

model_Conv_1 = make_mod(VOCAB_SIZE, WIN_SIZE, 3)

mymodel = compile_train_eval_model(
    model_Conv_1,
    x_train, y_train,
    x_test, y_test,
    optimizer='adam',
    epochs=15,
    batch_size=200,
    class_labels=CLASS_LIST,
    title='elonmask tweets'
)

y_pred = mymodel.predict(x_val)

print(y_pred)
print(y_val)

# Разберем результаты предсказания

r = np.argmax(y_pred, axis=1)
unique, counts = np.unique(r, return_counts=True)
counts = counts/y_pred.shape[0]*100
print(unique, counts)

