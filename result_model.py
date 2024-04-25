# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import minimize
import tensorflow_probability as tfp
import pandas as pd

# Смешанная точность
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Accelerated Linear Algebra
tf.config.optimizer.set_jit(True)

# Функции для расчета новых признаков
def calculate_incidence_angle(Ux, Uy, Uz):
    normal_vector = np.array([0, 0, 1])
    velocity_vector = np.array([Ux, Uy, Uz])
    unit_normal = normal_vector / np.linalg.norm(normal_vector)
    unit_velocity = velocity_vector / np.linalg.norm(velocity_vector)
    dot_product = np.dot(unit_normal, unit_velocity)
    incidence_angle = np.arccos(dot_product)
    return incidence_angle

def calculate_kinetic_energy(Ux, Uy, Uz, mass):
    velocity_squared = Ux**2 + Uy**2 + Uz**2
    kinetic_energy = 0.5 * mass * velocity_squared
    return kinetic_energy

def calculate_momentum(Ux, Uy, Uz, mass):
    momentum = mass * np.sqrt(Ux**2 + Uy**2 + Uz**2)
    return momentum

# Функция загрузки и подготовки данных
def load_dataset(filename):
    data = pd.read_csv(filename, sep=' ', header=None)
    X_base = data.iloc[:, [1, 3, 4, 5]]  # время залипания, Uвх
    X = pd.DataFrame()
    X["time"] = X_base.iloc[:, 0]
    X["Ux"] = X_base.iloc[:, 1]
    X["Uy"] = X_base.iloc[:, 2]
    X["Uz"] = X_base.iloc[:, 3]
    # Добавляем новые признаки
    X["incidence_angle"] = calculate_incidence_angle(X["Ux"], X["Uy"], X["Uz"])
    X["kinetic_energy"] = calculate_kinetic_energy(X["Ux"], X["Uy"], X["Uz"], mass= 67 * 1e-27)  # mass=1 - пример, замените на реальную массу
    X["momentum"] = calculate_momentum(X["Ux"], X["Uy"], X["Uz"], mass= 67 * 1e-27)
    y = data.iloc[:, 6:9]  # Uвых

    X = X.iloc[:60000]
    y = y.iloc[:60000]
    return X.values, y.values

# Определение архитектуры модели (пример с LSTM)
class LSTM_Model(keras.Model):
    def __init__(self, num_features, num_neurons, num_outputs):
        super(LSTM_Model, self).__init__()
        self.lstm_layer = layers.LSTM(num_neurons)
        self.dense_layer = layers.Dense(num_outputs)

    def call(self, inputs):
        x = self.lstm_layer(inputs)
        outputs = self.dense_layer(x)
        return outputs

# Определение функций потерь
def data_loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)

def physics_loss(y_true, y_pred):
      # Параметры (замените на реальные значения)
    mass = 67 * 1e-27
    coefficient_of_restitution = 0.8  # Коэффициент восстановления для неупругого столкновения

    # Импульс до и после столкновения
    momentum_before = mass * y_true[:, :3]
    momentum_after = mass * y_pred

    # Кинетическая энергия до и после столкновения
    kinetic_energy_before = 0.5 * mass * tf.reduce_sum(tf.square(y_true[:, :3]), axis=1)
    kinetic_energy_after = 0.5 * mass * tf.reduce_sum(tf.square(y_pred), axis=1)

    # Потеря энергии
    energy_loss = tf.abs(kinetic_energy_before - coefficient_of_restitution * kinetic_energy_after)

    # Потеря импульса
    momentum_loss = tf.reduce_sum(tf.abs(momentum_before - momentum_after), axis=1)

    # Комбинированная потеря
    total_loss = energy_loss + momentum_loss
    return tf.reduce_mean(total_loss)

# Загрузка и подготовка данных
X, y = load_dataset("Ar_300_rough_collision.dat")  # Замените на имя вашего файла
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
X_train = tf.expand_dims(X_train, axis=1)
X_test = tf.expand_dims(X_test, axis=1)

# Создание и обучение модели
model = LSTM_Model(num_features=X_train.shape[1], num_neurons=50, num_outputs=3)

model.compile(loss=[data_loss, physics_loss], loss_weights=[1.0, 0.1], optimizer=keras.optimizers.Adam(learning_rate=7e-3))
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f"loss={loss}")
