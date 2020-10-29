import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("20201020210709-summary.csv")
plt.bar(df.hidden_layer_size, df.roc_auc_mean)
plt.show()


# for y in df["loss_func"].unique().tolist():
#     for x in df["opt_func"].unique().tolist():
#         for _, row in df[(df["opt_func"] == x) & (df["loss_func"] == y)].iterrows():
#             plt.plot([0, 1], [0, 1], linestyle="--", label="")
#             # print(row["hidden_layer_size"])
#             # print(row["fprs_mean"])
#             # print(row["tprs_mean"])
#             fpr = eval(row["fprs_mean"])
#             tpr = eval(row["tprs_mean"])
#             # print(type(fpr))
#             h_size = row["hidden_layer_size"]
#             l_length = row["layers_length"]
#             opt_func = row["opt_func"]
#             loss_func = row["loss_func"]
#             lbl = f"{h_size}x{l_length} - {loss_func} Loss - {opt_func} Opt"
#             plt.plot(fpr, tpr, linestyle="-", label=lbl)

#         plt.legend()
#         plt.show()
