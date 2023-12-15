from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from config.dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by, plot_multiline_chart, autocorrelation_study, eval_stationarity
from numpy import array
from matplotlib.pyplot import show, subplots, plot, legend
from matplotlib.figure import Figure
from config.dslabs_functions import set_chart_labels
from statsmodels.tsa.stattools import adfuller


file_tag = "covid"
target = "deaths"
data: DataFrame = read_csv(
    "forecasting_health/data/forecast_covid_single.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]


deaths_quarters: Series = ts_aggregation_by(series, "Q")
figure(figsize=(3 * HEIGHT, HEIGHT ))
plot_line_chart(
    deaths_quarters.index.to_list(),
    deaths_quarters.to_list(),
    xlabel="quarters",
    ylabel=target,
    title=f"{file_tag} quarterly mean {target}",
)
show()
savefig("forecasting_health/profiling/images/covid_quarterly_reading.png")

deaths_months: Series = ts_aggregation_by(series, "M")
figure(figsize=(3 * HEIGHT, HEIGHT ))
plot_line_chart(
    deaths_months.index.to_list(),
    deaths_months.to_list(),
    xlabel="months",
    ylabel=target,
    title=f"{file_tag} monthly mean {target}",
)
show()
savefig("forecasting_health/profiling/images/covid_monthly_reading.png")

fig: Figure
axs: array
fig, axs = subplots(2, 3, figsize=(2 * HEIGHT, HEIGHT))
set_chart_labels(axs[0, 0], title="WEEKLY")
axs[0, 0].boxplot(series)

set_chart_labels(axs[0, 1], title="MONTHLY")
axs[0, 1].boxplot(deaths_months)
set_chart_labels(axs[0, 2], title="QUARTER")
axs[0, 2].boxplot(deaths_quarters)


axs[1, 0].grid(False)
axs[1, 0].set_axis_off()
axs[1, 0].text(0.2, 0, str(series.describe()), fontsize="small")


axs[1, 1].grid(False)
axs[1, 1].set_axis_off()
axs[1, 1].text(0.2, 0, str(deaths_months.describe()), fontsize="small")

axs[1, 2].grid(False)
axs[1, 2].set_axis_off()
axs[1, 2].text(0.2, 0, str(deaths_quarters.describe()), fontsize="small")

show()
savefig("forecasting_health/profiling/images/covid_boxplots_forecast.png")

grans: list[Series] = [series, deaths_months, deaths_quarters]
gran_names: list[str] = ["Weekly", "Monthly", "Quarterly"]
fig: Figure
axs: array
fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
fig.suptitle(f"{file_tag} {target}")
for i in range(len(grans)):
    set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=target, ylabel="Nr records")
    axs[i].hist(grans[i].values)
show()
savefig("forecasting_health/profiling/images/covid_histograms_forecast.png")




def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series


figure(figsize=(3 * HEIGHT, HEIGHT))
lags = get_lagged_series(series, 20, 10)
plot_multiline_chart(series.index.to_list(), lags, xlabel="date", ylabel=target)

show()
savefig("forecasting_health/profiling/images/covid_histograms_forecast_lag_plot.png")

autocorrelation_study(series, 10, 1)

show()
savefig("forecasting_health/profiling/images/covid_autocorrelation_study.png")



BINS = 10
mean_line: list[float] = []
n: int = len(series)

for i in range(BINS):
    segment: Series = series[i * n // BINS : (i + 1) * n // BINS]
    mean_value: list[float] = [segment.mean()] * (n // BINS)
    mean_line += mean_value
mean_line += [mean_line[-1]] * (n - len(mean_line))

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} stationary study",
    name="original",
    show_stdev=True,
)
n: int = len(series)
plot(series.index, mean_line, "r-", label="mean")
legend()
show()
savefig("forecasting_health/profiling/images/covid_stationary_study.png")







print(f"The series {('is' if eval_stationarity(series) else 'is not')} stationary")
