import json
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from straikit.eval.metrics.generate_metrics import generate_time_series_metrics
from straikit.preprocessing.manipulation.ts_dataset import TSWindowedXY
from straikit.preprocessing.normalization.Standard_Scaler import StandardScaler
from trsf_iter_pred import trsf_iter_predict
from ts import TSTransformer

data_file = "bank"
MAX_PLOT = 300
PRED_STEP = 24
EPOCHS = 48
DO_SCALE = True

if data_file == "elec":
    data = pd.read_csv("~/projects/deepglo/datasets/electricity.csv")
    data["0"] = data["17"].shift(-4)
    data["0"].ffill(inplace=True)
    # feature = data[["0"]]
    feature = data[["2", "3", "4", "5", "6", "7", "9", "10", "11", "12"]]
    series = data["17"]
    feature = feature.values
    series = series.to_numpy()

elif data_file == "bank":
    data = pd.read_csv(
        "~/projects/ts_dataset/client/Bank_Customer_Walkin_Forecasting_Mod.csv",
    )
    feature = data[["temperature", "weather", "workingday", "holiday", "season"]]
    series = data[["Bank Customer Walkin count"]]
    feature = feature.values
    series = series.to_numpy()

elif data_file == "traffic":
    data = pd.read_csv("~/projects/deepglo/datasets/traffic.csv")
    data["0"] = data["8"].shift(-2)
    data["0"].ffill(inplace=True)
    # feature = data[["0"]]
    feature = data[["2", "3", "4", "5", "6", "7", "9", "10", "11", "12", "13"]]
    series = data["13"]
    feature = feature.values
    series = series.to_numpy()

else:
    raise ValueError("Data file not identified!")

ltrain = int(len(series) * 0.8)

train_data = series[:ltrain]
test_data = series[ltrain:]

original_test = deepcopy(test_data)

if DO_SCALE:
    mu = np.mean(train_data)
    sigma = np.std(train_data)
    train_data = (train_data - mu) / sigma
    test_data = (test_data - mu) / sigma



t2v_size = 4
embed_size = 12
num_layers = 3
d_model = 16
num_heads = 4
dff = 16
rate = 0.0
window_size = 64
flex_embed = False

xy = TSWindowedXY(window_size=window_size, use_feature=False)

print("Data loaded +++++++++++++++++++++++++")
train_xy = xy.fit_transform(X=None, y=train_data)
train_x = np.array([x[0] for x in train_xy["X"]])
train_y = np.array([y[0] for y in train_xy["y"]])

print("Train and test data prepared ++++++++++++++++++++++++++++++++++")


m = TSTransformer(
    t2v_size=t2v_size,
    embed_size=embed_size,
    window_size=window_size,
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    rate=rate,
    flex_embed=flex_embed,
)

m.compile(optimizer="adam", loss="mae")
print("fitting started .................")
m.fit(train_x, train_y, epochs=EPOCHS, batch_size=64)
m.summary()

print("++++++++++++++++++++++++++++++Training finished++++++++++++++++++++++++++++++++")

sid = max(0, len(test_data) - MAX_PLOT)
error1 = []
error2 = []


plt.plot(np.array(original_test[sid:]), "r", label="test_data")
for step in range(PRED_STEP):
    dl_iter_result = trsf_iter_predict(
        m,
        test_data,
        do_diff=False,
        use_feature=False,
        window_size=window_size,
        test_feature=None,
        pred_step=step + 1,
        d_model=d_model,
    )
    i_pred = dl_iter_result
    if DO_SCALE:
        i_pred = i_pred * sigma + mu
    if step == PRED_STEP - 1:
        plt.plot(np.array(i_pred[sid:]), label=f"dl_prediction")
    metrics = generate_time_series_metrics(
        y_pred=i_pred,
        y_true=original_test,
        metrics={"mse", "smape", "msse", "mae"},
    )
    print(f"Metric for iter step {step + 1}: {metrics.values[0][0]}")
    error2.append(json.loads(metrics.values[0][0])["msse"])

plt.legend()
plt.show()
