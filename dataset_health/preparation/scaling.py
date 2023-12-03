# Imports
import pandas as pd

from dslabs_functions import (
    get_variable_types,plot_multibar_chart,
    mvi_by_filling,
    evaluate_approach)

from numpy import ndarray
from pandas import DataFrame, read_csv,Series
from matplotlib.pyplot import savefig, show, figure, subplots
from pandas import read_csv, DataFrame, Series
from dslabs_functions import (
    NR_STDEV,
    get_variable_types,
    determine_outlier_thresholds_for_var,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# read class_pos_covid_derived_prepared
filename = "dataset_health/data/class_pos_covid_derived_prepared.csv"
data = pd.read_csv(filename, sep=',', decimal='.', na_values='')
df_mv: DataFrame = mvi_by_filling(data, strategy="frequent")

target = 'CovidPos'
train, test = train_test_split(df_mv, test_size=0.2)

n_std: int = NR_STDEV
numeric_vars: list[str] = get_variable_types(train)["numeric"]
if numeric_vars is not None:
    df: DataFrame = train.copy(deep=True)
    summary5: DataFrame = train[numeric_vars].describe()
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
            summary5[var]
        )
        outliers: Series = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)
else:
    print("There are no numeric variables")


vars: list[str] = df.columns.to_list()
target_data: Series = df.pop(target)
transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
    df
)
df_zscore = DataFrame(transf.transform(df), index=df.index)
df_zscore[target] = target_data
# df_zscore.columns = vars


transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df)
df_minmax = DataFrame(transf.transform(df), index=df.index)
df_minmax[target] = target_data
# df_minmax.columns = vars


fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
axs[0, 1].set_title("Original data after Outlier removal")
df.boxplot(ax=axs[0, 0])
axs[0, 0].set_title("Z-score normalization")
df_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title("MinMax normalization")
df_minmax.boxplot(ax=axs[0, 2])
# show()
fig.savefig('dataset_health/preparation/scaling_images/normalized_data_boxplots.png')

figure()
eval: dict[str, list] = evaluate_approach(df_zscore, test.copy(deep=True), target=target, metric="recall", estimators_names= ["GaussianNB", "BernoulliNB"])
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Covid zscore normalization evaluation", percentage=True
)
savefig(f"dataset_health/preparation/scaling_images/covid_zscore_norm_eval.png")

figure()
eval: dict[str, list] = evaluate_approach(df_minmax, test.copy(deep=True), target=target, metric="recall", estimators_names= ["GaussianNB", "BernoulliNB"])
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Covid minmax normalization evaluation", percentage=True
)
savefig(f"dataset_health/preparation/scaling_images/covid_minmax_norm_eval.png")


