import numpy as np
from straikit.preprocessing.manipulation.ts_dataset import TSWindowedXY
from straikit.supervized.forecasting.prediction_loop import PredictionLoop, reconstruct
from straikit.preprocessing.manipulation.diff_transform import DiffTransformer


def ts_inference_padding(y_true, y_pred, forecasting_horizon: int = 1):
    forecast = [np.nan] * (len(y_true) + forecasting_horizon)
    window_size = len(y_true) - len(y_pred) + 1
    if len(y_pred) > 1:
        forecast[window_size + forecasting_horizon - 1 :] = y_pred.squeeze()
    else:
        forecast[-1] = y_pred[0]
    y_pred = np.array(forecast)

    return y_pred


def trsf_iter_predict(
    estimator,
    test_data,
    do_diff,
    use_feature,
    window_size,
    pred_step,
    d_model,
    test_feature=None,
):
    if do_diff:
        df = DiffTransformer()
        df.fit(test_data)
        if use_feature:
            diff_test_feature = df.transform(test_feature)
        else:
            diff_test_feature = None
        diff_test_series = df.transform(test_data)
    else:
        diff_test_series = test_data
        if use_feature:
            diff_test_feature = test_feature
        else:
            diff_test_feature = None

    xy = TSWindowedXY(
        window_size=window_size, use_feature=use_feature, prediction_step=pred_step
    )

    if do_diff:
        test_x = xy.inf_transform(X=diff_test_feature, y=diff_test_series)
    else:
        test_x = xy.inf_transform(X=test_feature, y=test_data)

    test_x_to_array = np.array([x[0] for x in test_x["X"]])
    step_in = np.copy(test_x_to_array)

    pp = PredictionLoop(
        do_diff=do_diff, use_feature=use_feature, prediction_step=pred_step
    )
    pp.fit([], test_data)

    if do_diff:
        final_result = test_data
        for step in range(pred_step):
            step_out = estimator(step_in, training=False)
            if step < pred_step - 1:
                step_in = pp.transform(
                    step_in,
                    step_out,
                    feature=diff_test_feature,
                    window_size=window_size,
                    step_count=step,
                )
            final_result = reconstruct(
                final_result,
                step_out,
                use_feature=use_feature,
                first_step=(step == 0),
                prediction_step=pred_step,
            )

    else:
        out_put = None
        for step in range(pred_step):
            # print(estimator(step_in, training=False).numpy())
            step_out = estimator(step_in, training=False).numpy()[:, -1:, 0:]
            if step < pred_step - 1:
                step_in = np.concatenate(
                    (step_in[:, 1:, :], step_out), axis=1
                )
            else:
                out_put = step_out
        final_result = ts_inference_padding(
            test_data, out_put, forecasting_horizon=1 if use_feature else pred_step
        )

    return final_result
