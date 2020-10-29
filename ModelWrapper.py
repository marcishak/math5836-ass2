import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow import keras

class ModelWrapper:
    """
    Wrapper for defining and testing Keras Models
    """

    def __repr__(self):
        try:
            stringlist = []
            self.model.summary(print_fn=lambda x: stringlist.append(x))
            short_model_summary = "\n".join(stringlist)
            return short_model_summary
        except AttributeError:
            return "Model is still undefined / has not been built"

    def __init__(self, name=None, inputs=None, h_layers=None, outputs=None):
        """
        Keras model initalizer
        """
        self.input = inputs
        self.h_layers = h_layers
        self.outputs = outputs
        self.name = name

    def build_model(self):
        """
        Wrapper for bulding the model with keras.Model
        """
        self.outputs = self.outputs(self.h_layers)
        self.model = keras.Model(
            inputs=self.input, outputs=self.outputs, name=self.name
        )

    def compile_model(self, loss, optimizer, metrics):
        """
        Wrapper for compiling the model with model.compile
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def plot_model(self, path="data/reporting/model_plot.png", show_shapes=True):
        """
        Wrapper for keras.utils.plot_model
        """
        keras.utils.plot_model(self.model, path, show_shapes=show_shapes)
        plt.clf()

    def model_summary(self, path=None):
        """
        Prints model summary to path
        """
        if path is None:
            print(self.model.summary())
        else:
            # from:
            # https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
            # Open the file
            with open(path, "w") as fh:
                # Pass the file handle in as a lambda function to make it callable
                self.model.summary(print_fn=lambda x: fh.write(x + "\n"))

    def add_layer(self, layer):
        """
        adds layer to model
        """
        # self.plot_model()
        # print(self.input)
        if self.h_layers is not None:
            # print(self.h_layers)
            self.h_layers = layer(self.h_layers)
            # print(self.h_layers)
        else:
            # print(self.input)
            self.h_layers = layer(self.input)
            # print(self.input)

    def fit(self, X, y, **kwargs):
        """
        TODO: docstring
        """
        self.fit_history = self.model.fit(X, y, **kwargs)

    def predict(self, X):
        """
        wrapper for model.predict
        """
        return self.model.predict(X)

    def predict_classes(self, X):
        return self.model.predict_classes(X)

    def plot_epoch_loss(self, hist_test="loss", label="Loss", path=None):
        """
        wrapper for plotting epoch loss
        """
        # based off https://chrisalbon.com/deep_learning/keras/visualize_loss_history/
        training_loss = self.fit_history.history[hist_test]
        test_loss = self.fit_history.history["val_" + hist_test]
        epoch_count = range(1, len(training_loss) + 1)
        plt.plot(epoch_count, training_loss, "r--")
        plt.plot(epoch_count, test_loss, "b-")
        plt.legend(["Training " + label, "Test " + label])
        plt.xlabel("Epoch")
        plt.ylabel(label)
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
        plt.clf()

    # def add_multiple_layers(self, count, layers):
    #     """
    #     adds layers from list of layers count times
    #     """
    #     for _ in range(count):
    #         for i in range(len(layers)):
    #             print(i)
    #             self.add_layer(layers[i])


def fit_plot_roc(mod, X_test, y_test, path=None):
    """
    TODO: docstring
    """
    y_preds = mod.predict(X_test)
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
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    return f"roc_auc_score: {roc_auc_score(y_test, y_preds)}"
