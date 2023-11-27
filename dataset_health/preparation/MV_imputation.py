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
df: DataFrame = mvi_by_filling(data, strategy="frequent")

target = 'CovidPos'
train, test = train_test_split(df, test_size=0.2)

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Covid MV frequent evaluation", percentage=True
)
savefig(f"dataset_health/images/covid_mv_frequent_eval.png")

# read class_pos_covid_derived_prepared
filename = 'dataset_health/class_pos_covid_derived_prepared.csv'
data = pd.read_csv(filename, sep=',', decimal='.', na_values='')
df: DataFrame = mvi_by_filling(data, strategy="mean")

target = 'CovidPos'
train, test = train_test_split(df, test_size=0.2)

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Covid MV mean evaluation", percentage=True
)
savefig(f"dataset_health/images/covid_mv_mean_eval.png")

