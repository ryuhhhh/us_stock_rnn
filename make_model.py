"""
RNNで学習します
"""
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras

def custom_total_true(y_true, y_pred):
    return K.sum(y_true)

def custom_total_pred(y_true, y_pred):
    return K.sum(y_pred)

def custom_recall(y_true, y_pred):
    """クラス1のRecall"""
    true_positives = K.sum(y_true * y_pred)
    total_positives = K.sum(y_true)
    return true_positives / (total_positives + K.epsilon())

def custom_precision(y_true, y_pred):
    """クラス1のPrecision"""
    total_1_predictions = K.sum(y_pred)
    total_true_predictions = K.sum(y_true*y_pred)
    return total_true_predictions/(total_1_predictions+K.epsilon())

if __name__ == '__main__':
    model = keras.models.Sequential([
        # input_shapeは10時系列 1次元(数値)
        keras.layers.GRU(10,return_sequenses=True,input_shape=(10,1)),
        keras.layers.GRU(10,return_sequenses=True),
        keras.layers.Dense(1,activation='sigmoid')
    ])

    # optimizer = keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpoint = keras.callbacks.ModelCheckpoint('./drive/MyDrive/Colab Notebooks/model1day.h5')
    early_stopping = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=30,
        validation_data=(X_valid,y_valid),
        class_weight=class_weight,
        callbacks=[checkpoint,early_stopping]
    )