from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from config.dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by, plot_multiline_chart, autocorrelation_study, eval_stationarity, plot_components
from numpy import array, arange
from matplotlib.pyplot import show, subplots, plot, legend
from matplotlib.figure import Figure
from config.dslabs_functions import set_chart_labels, plot_line_chart, series_train_test_split, plot_forecasting_eval, plot_forecasting_series
from statsmodels.tsa.stattools import adfuller
from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, show
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from pandas import Series
from sklearn.base import RegressorMixin


from sklearn.base import RegressorMixin


from config.dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_multiline_chart


def arima_study(train: Series, test: Series, measure: str = "R2"):
    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}
        for q in q_params:
            yvalues = []
            for p in p_params:
                arima = ARIMA(train, order=(p, d, q))
                model = arima.fit()
                prd_tst = model.forecast(steps=len(test), signal_only=False)
                eval: float = FORECAST_MEASURES[measure](test, prd_tst)
                # print(f"ARIMA ({p}, {d}, {q})", eval)
                if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                    best_performance: float = eval
                    best_params["params"] = (p, d, q)
                    best_model = model
                yvalues.append(eval)
            values[q] = yvalues
        plot_multiline_chart(
            p_params, values, ax=axs[i], title=f"ARIMA d={d} ({measure})", xlabel="p", ylabel=measure, percentage=flag
        )
    print(
        f"ARIMA best results achieved with (p,d,q)=({best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}) ==> measure={best_performance:.2f}"
    )

    return best_model, best_params


def scale_all_dataframe(data: DataFrame) -> DataFrame:
    vars: list[str] = data.columns.to_list()
    transf: StandardScaler = StandardScaler().fit(data)
    df = DataFrame(transf.transform(data), index=data.index)
    df.columns = vars
    return df

file_tag = 'covid'
timecol = 'date'
target = 'deaths'
measure = 'R2'

data: DataFrame = read_csv(
    "forecasting_health/data/forecast_covid_single.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

ss_diff: Series = data.diff().diff()
ss_diff = scale_all_dataframe(ss_diff)
train, test = series_train_test_split(ss_diff, trn_pct=0.90)


trnX = arange(len(train)).reshape(-1, 1)
trnY = train.to_numpy()
tstX = arange(len(train), len(ss_diff)).reshape(-1, 1)
tstY = test.to_numpy()
trnY[0] = 0
trnY[1] = 0


predictor = ARIMA(train, order=(3, 1, 2))
model = predictor.fit()
print(model.summary())

model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
savefig(f"forecasting_health/forecasting/images/{file_tag}_arima_diagnostic.png")

best_model, best_params = arima_study(train, test, measure=measure)
savefig(f"forecasting_health/forecasting/images/{file_tag}_arima_{measure}_study.png")

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

plot_forecasting_eval(
    train, test, prd_trn, prd_tst, title=f"{file_tag} - ARIMA (p={params[0]}, d={params[1]}, q={params[2]})"
)
savefig(f"forecasting_health/forecasting/images/{file_tag}_arima_{measure}_eval.png")



plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - ARIMA ",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"forecasting_health/forecasting/images/{file_tag}_arima_{measure}_forecast.png")