import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from datetime import datetime

import ModelInstance as mw

X_train = np.genfromtxt("data/cleaned/park_X_train.csv", delimiter=",")
y_train = np.genfromtxt("data/cleaned/park_y_train.csv", delimiter=",")
X_test = np.genfromtxt("data/cleaned/park_X_test.csv", delimiter=",")
y_test = np.genfromtxt("data/cleaned/park_y_test.csv", delimiter=",")

# Bootstrap
# ind = np.random.choice(range(X_train.shape[1]), replace=True, size=10000)
# X_train = X_train[ind, :]
# y_train = y_train[ind]


def train_model_grid(ziplayers, loss_functions, opt_functions, instance_count=10):

    full_summary_df = pd.DataFrame()
    for layers_list, hidden_layer_size in ziplayers:
        for layers in layers_list:
            for loss_func in loss_functions:
                for opt_func in opt_functions:
                    model_runs = []
                    for _ in range(instance_count):
                        mi = mw.ModelInstance(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            layers,
                            metrics=["AUC", "acc"],
                            output_activation="sigmoid",
                            loss_func=loss_func,
                            opt_func=opt_func,
                            validation_split=0.5,
                        )
                        mi.fit_predict_model("validation", epochs=300, batch_size=300)
                        mi.build_classifcation_report()
                        model_runs.append(mi)
                    summary_df = mw.summarise_model_instances(model_runs, hidden_layer_size, len(layers))
                    full_summary_df = full_summary_df.append(
                        pd.DataFrame(summary_df), ignore_index=True,
                    )
    return full_summary_df


def build_layers(hidden_layers):
    """
    wish we had lazy eval here
    """
    layers = [
        [
            [
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                )
            ],
            [
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
            ],
            [
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
            ],
        ]
        for x in hidden_layers
    ]
    return zip(layers, hidden_layers)


test = build_layers(
    [
        1,
        5,
        10,
        50,
    ]
)


summary_grid = train_model_grid(
    test,
    [
        keras.losses.KLDivergence(),
        keras.losses.Hinge(),
        keras.losses.BinaryCrossentropy(),
    ],
    [
        keras.optimizers.SGD(),
        keras.optimizers.Adam(),
        keras.optimizers.RMSprop(),
    ],
)

dt = datetime.now().strftime("%Y%m%d%H%M%S")
summary_grid.to_csv(f"{dt}-summary.csv")
