from config.dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart
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

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.preprocessing import StandardScaler
from pandas import Series
from sklearn.base import RegressorMixin
from matplotlib.pyplot import savefig



from sklearn.base import RegressorMixin



def exponential_smoothing_study(train: Series, test: Series, measure: str = "R2"):
    alpha_values = [i / 10 for i in range(1, 10)]
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Exponential Smoothing", "metric": measure, "params": ()}
    best_performance: float = -100000

    yvalues = []
    for alpha in alpha_values:
        tool = SimpleExpSmoothing(train)
        model = tool.fit(smoothing_level=alpha, optimized=False)
        prd_tst = model.forecast(steps=len(test))

        eval: float = FORECAST_MEASURES[measure](test, prd_tst)
        # print(w, eval)
        if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["params"] = (alpha,)
            best_model = model
        yvalues.append(eval)

    print(f"Exponential Smoothing best with alpha={best_params['params'][0]:.0f} -> {measure}={best_performance}")
    plot_line_chart(
        alpha_values,
        yvalues,
        title=f"Exponential Smoothing ({measure})",
        xlabel="alpha",
        ylabel=measure,
        percentage=flag,
    )

    return best_model, best_params





def scale_all_dataframe(data: DataFrame) -> DataFrame:
    vars: list[str] = data.columns.to_list()
    transf: StandardScaler = StandardScaler().fit(data)
    df = DataFrame(transf.transform(data), index=data.index)
    df.columns = vars
    return df

measure: str = "R2"
file_tag = 'covid'
timecol = 'date'
target = 'deaths'

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



best_model, best_params = exponential_smoothing_study(train, test, measure=measure)
savefig(f"forecasting_health/forecasting/images/{file_tag}_exponential_smoothing_{measure}_study.png")

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Exponential Smoothing alpha={params[0]}")
savefig(f"forecasting_health/forecasting/images/{file_tag}_exponential_smoothing_{measure}_eval.png")


plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Exponential Smoothing ",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"forecasting_health/forecasting/images/{file_tag}_exponential_smoothing_{measure}_forecast.png")