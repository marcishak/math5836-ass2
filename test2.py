# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def array(x):
    return x


# %%
df = pd.read_csv("20201029020350-summary.csv")


# %%
df


# %%
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(df.loss_func, df.roc_auc_scores)
plt.show()


# %%

for x in df.layers_length.unique().tolist():
    plt.clf()
    plt.plot([0, 1], [0, 1], linestyle="--", label="")
    i = 0
    for _, row in df[df["layers_length"] == x].iterrows():
        if i % 2 == 0:
            # print(row["hidden_layer_size"])
            # print(row["fprs_mean"])
            # print(row["tprs_mean"])
            fpr = eval(row["fprs"])[0]
            tpr = eval(row["tprs"])[0]
            # print(type(fpr))
            h_size = row["hidden_layer_size"]
            lbl=f"{h_size} Hidden Layer Size"
            print(lbl)
            print(fpr)
            print(tpr)
            plt.plot(
                fpr,
                tpr,
                linestyle="-",
                label=lbl
            )
        i += 1
    tlt = f"ROC of {x} layer NN"
    plt.title(tlt)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


# %%

plt.plot([0, 1], [0, 1], linestyle="--", label="")
for _, row in df.iterrows():
    # print(row["hidden_layer_size"])
    # print(row["fprs_mean"])
    # print(row["tprs_mean"])
    fpr = eval(row["fprs"])[0]
    tpr = eval(row["tprs"])[0]
    # print(type(fpr))
    h_size = row["hidden_layer_size"]
    l_count = row["layers_length"]
    lbl=f"{h_size} Hidden Layer Size - {l_count} layer(s)"
    plt.plot(
        fpr,
        tpr,
        linestyle="-",
        label=lbl
    )
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# %%



