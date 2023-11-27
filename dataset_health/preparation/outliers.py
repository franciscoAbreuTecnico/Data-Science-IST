# Imports
import pandas as pd

from dslabs_functions import (
    get_variable_types,plot_multibar_chart,
    mvi_by_filling,
    evaluate_approach)


from numpy import ndarray
from pandas import DataFrame, read_csv,Series
from matplotlib.pyplot import savefig, show, figure
from pandas import read_csv, DataFrame, Series
from dslabs_functions import (
    NR_STDEV,
    get_variable_types,
    determine_outlier_thresholds_for_var,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read class_pos_covid_derived_prepared
filename = 'dataset_health/class_pos_covid_derived_prepared.csv'
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



figure()
eval: dict[str, list] = evaluate_approach(df, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Covid drop outliers evaluation", percentage=True
)
savefig(f"dataset_health/images/covid_drop_out_eval.png")


filename = 'dataset_health/class_pos_covid_derived_prepared.csv'
data = pd.read_csv(filename, sep=',', decimal='.', na_values='')
df_mv: DataFrame = mvi_by_filling(data, strategy="frequent")

target = 'CovidPos'
train, test = train_test_split(df_mv, test_size=0.2)



if [] != numeric_vars:
    df: DataFrame = train.copy(deep=True)
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        median: float = df[var].median()
        df[var] = df[var].apply(lambda x: median if x > top or x < bottom else x)
else:
    print("There are no numeric variables")


figure()
eval: dict[str, list] = evaluate_approach(df, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Covid replace outliers evaluation", percentage=True
)
savefig(f"dataset_health/images/covid_replace_out_eval.png")




