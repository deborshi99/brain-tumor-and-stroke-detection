from analysis import X_train, X_val, y_val, y_train, y_test, X_test
import tensorflow as tf
from keras.layers import Dropout, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras import Sequential
from keras.activations import relu, sigmoid
from sklearn.metrics import classification_report


def model2():
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train.shape[1],), activation=relu))
    model.add(Dense(256, activation=relu))
    model.add(Dense(256, activation=relu))
    model.add(Dense(512, activation=relu))
    model.add(Dense(1, activation=sigmoid))

    model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.003), metrics=["accuracy"]) # best results at 0.003

    return model

model2 = model2()
history2 = model2.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    shuffle=True
)

y_pred_2 = model2.predict(X_test)
y_pred_2 = tf.round(y_pred_2)
print(classification_report(y_pred_2, y_test))

model2.save("model/main_model.h5")