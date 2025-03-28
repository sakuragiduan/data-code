# -*- coding: utf-8 -*-
import os
import numpy as np
from datetime import datetime
from tensorflow import keras


class DQN:
    def __init__(self, state_size):
        self.state_size = state_size
        self.gamma = 0.95  # discount rate
        self.model = self._build_model()
        self.epochs = 1
        self.batch_size = 1024

    def _build_model(self):
        keras_get = lambda x, pos: x[:, pos]
        keras_slice = lambda x, pos: x[:, pos:]
        x = keras.layers.Input(shape=(self.state_size,))

        x_1 = keras.layers.Lambda(keras_get, arguments={'pos': 0})(x)
        x_1 = keras.layers.Embedding(225, 10, embeddings_initializer='uniform')(x_1)
        x_2 = keras.layers.Lambda(keras_get, arguments={'pos': 1})(x)
        x_2 = keras.layers.Embedding(3600, 60, embeddings_initializer='uniform')(x_2)
        x_3 = keras.layers.Lambda(keras_slice, arguments={'pos': 2})(x)

        x_0 = keras.layers.concatenate([x_1, x_2, x_3])

        x_1 = keras.layers.Dense(256, activation='relu')(x_0)
        x_2 = keras.layers.Dense(168, activation='relu')(x_1)
        x_3 = keras.layers.Dense(64, activation='relu')(x_2)
        x_4 = keras.layers.Dense(32, activation='relu')(x_3)
        y = keras.layers.Dense(1)(x_4)

        model = keras.Model(inputs=x, outputs=y)
        model.compile(loss='mse', optimizer='adam')

        return model

    def predict(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        v_value = self.model.predict(state)
        return v_value

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train_model(self, input_data: np, output_data: np, model_save_root_path: str, episode: int):
        if episode != 0:
            self.load(
                os.path.join(model_save_root_path, 'dqn_model.h5'))
        if (episode + 1) % 500 == 0:
            log_dir = os.path.join(model_save_root_path, 'observe', datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
            results = self.model.fit(input_data, output_data, epochs=self.epochs, batch_size=self.batch_size, verbose=0,
                                     callbacks=[tensorboard_callback])
        else:
            print('without tensorboard recording')
            results = self.model.fit(input_data, output_data, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        self.save(
            os.path.join(model_save_root_path, 'dqn_model.h5'))

        return np.array(results.history['loss'])

