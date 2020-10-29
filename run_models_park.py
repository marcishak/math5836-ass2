import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_curve
from tensorflow import keras
from tensorflow.keras import initializers, layers

X_train = np.genfromtxt("data/cleaned/park_X_train.csv", delimiter=",")
y_train = np.genfromtxt("data/cleaned/park_y_train.csv", delimiter=",")
X_test = np.genfromtxt("data/cleaned/park_X_test.csv", delimiter=",")
y_test = np.genfromtxt("data/cleaned/park_y_test.csv", delimiter=",")

ind = np.random.choice(range(X_train.shape[1]), replace=True, size = 10000)

X_train = X_train[ind, :]
y_train = y_train[ind]

h_layer_size = 5000
# print(X_train)

inputs = keras.Input(shape=(X_train.shape[1],))
h_layers = layers.Dense(h_layer_size, kernel_initializer="random_normal")(inputs)
h_layers = layers.Dense(
    h_layer_size, kernel_initializer="random_normal", activation="relu"
)(h_layers)
h_layers = layers.Dense(
    h_layer_size, kernel_initializer="random_normal", activation="relu"
)(h_layers)
h_layers = layers.Dense(
    h_layer_size, kernel_initializer="random_normal", activation="relu"
)(h_layers)
# h_layers = layers.Dense(h_layer_size, kernel_initializer="random_normal",
# activation="sigmoid")(h_layers)
# h_layers = layers.Dropout(rate=0.5)(h_layers)
# outputs = layers.Dense(1)(h_layers)
outputs = layers.Dense(1, activation="sigmoid")(h_layers)
# outputs = layers.Dense(1, activation="softmax")(h_layers)
model = keras.Model(inputs=inputs, outputs=outputs, name="Parkinsons")
print(model.summary())
keras.utils.plot_model(model, "data/reporting/my_first_model.png", show_shapes=True)

model.compile(
    # loss=keras.losses.BinaryCrossentropy(),
    loss=keras.losses.KLDivergence(),
    # loss=keras.losses.Hinge(),
    # optimizer=keras.optimizers.SGD(learning_rate = 0.0005, momentum=0.1),
    # optimizer=keras.optimizers.SGD(learning_rate = 0.0005, momentum=0.1),
    optimizer=keras.optimizers.Adam(learning_rate=0.000000001),
    # optimizer=keras.optimizers.RMSprop(momentum=0.9),
    # optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy", "AUC"],
)

model.fit(X_train, y_train)
y_preds = model.predict(X_test)
# y_preds = model.predict(X_train)
# print(y_preds)
# print(y_train)
# print(classification_report(y_test, y_preds))


fpr, tpr, _ = roc_curve(y_test, y_preds)
plt.plot([0, 1], [0, 1], linestyle="--", label="")
# calculate roc curve for model
# plot model roc curve
plt.plot(fpr, tpr, linestyle="-", label="Sigmoid Activation Output", alpha=0.8)
# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# show the legend
plt.legend()
# show the plot
plt.show()
