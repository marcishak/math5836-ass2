# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
df = pd.read_csv("full_testcsv.csv")


# %%
df


# %%
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.bar(df.hidden_layer_size, df.roc_auc_mean)
# plt.show()


# %%

for x in df.layers_length.unique().tolist():
    plt.clf()
    plt.plot([0, 1], [0, 1], linestyle="--", label="")
    for _, row in df[df["layers_length"] == x].iterrows():
        # print(row["hidden_layer_size"])
        # print(row["fprs_mean"])
        # print(row["tprs_mean"])
        fpr = eval(row["fprs_mean"])
        tpr = eval(row["tprs_mean"])
        # print(type(fpr))
        h_size = row["hidden_layer_size"]
        lbl = f"{h_size} Hidden Layer Size"
        plt.plot(fpr, tpr, linestyle="-", label=lbl)
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
    fpr = eval(row["fprs_mean"])
    tpr = eval(row["tprs_mean"])
    # print(type(fpr))
    h_size = row["hidden_layer_size"]
    l_count = row["layers_length"]
    lbl = f"{h_size} Hidden Layer Size - {l_count} layer(s)"
    plt.plot(fpr, tpr, linestyle="-", label=lbl)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
