# Imports
import pandas as pd
from pandas import read_csv, concat, DataFrame, Series
from imblearn.over_sampling import SMOTE


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
filename = 'dataset_health/data/class_pos_covid_derived_prepared.csv'
data = pd.read_csv(filename, sep=',', decimal='.', na_values='')
df_mv: DataFrame = mvi_by_filling(data, strategy="frequent")

target = 'CovidPos'
train, test = train_test_split(df_mv, test_size=0.2)

test.to_csv(f"dataset_health/data/test.csv", index=False)

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


#vars: list[str] = df.columns.to_list()
#target_data: Series = df.pop(target)
#transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
#    df
#)
#df_zscore = DataFrame(transf.transform(df), index=df.index)
#df_zscore[target] = target_data
# df_zscore.columns = vars



target_count: Series = df[target].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()

print("Minority class=", positive_class, ":", target_count[positive_class])
print("Majority class=", negative_class, ":", target_count[negative_class])
print(
    "Proportion:",
    round(target_count[positive_class] / target_count[negative_class], 2),
    ": 1",
)
values: dict[str, list] = {
    "Original": [target_count[positive_class], target_count[negative_class]]
}


df_positives: Series = df[df[target] == positive_class]
df_negatives: Series = df[df[target] == negative_class]


df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
df_under: DataFrame = concat([df_positives, df_neg_sample], axis=0)
df_under.to_csv(f"dataset_health/data/covid_under.csv", index=False)

print("Minority class=", positive_class, ":", len(df_positives))
print("Majority class=", negative_class, ":", len(df_neg_sample))
print("Proportion:", round(len(df_positives) / len(df_neg_sample), 2), ": 1")



RANDOM_STATE = 42

smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
y = df.pop(target).values
X: ndarray = df.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(df.columns) + [target]
df_smote.to_csv(f"dataset_health/data/covid_smote.csv", index=False)

smote_target_count: Series = Series(smote_y).value_counts()
print("Minority class=", positive_class, ":", smote_target_count[positive_class])
print("Majority class=", negative_class, ":", smote_target_count[negative_class])
print(
    "Proportion:",
    round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
    ": 1",
)
print(df_smote.shape)


target_data: Series = test.copy(deep=True).pop(target)
#scaled_test: DataFrame = DataFrame(transf.transform(test), index=test.index)
#scaled_test[target] = target_data
#scaled_test_copy: DataFrame = scaled_test.copy(deep=True)
#scaled_test_copy2: DataFrame = scaled_test.copy(deep=True)
test_copy = test.copy(deep=True)
test_copy2 = test.copy(deep=True)

df_pos_sample: DataFrame = DataFrame(
    df_positives.sample(len(df_negatives), replace=True)
)
df_over: DataFrame = concat([df_pos_sample, df_negatives], axis=0)

figure()
eval: dict[str, list] = evaluate_approach(df_over, test_copy, target=target, metric="recall", estimators_names= ["GaussianNB", "BernoulliNB"])
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Covid over evaluation", percentage=True
)
savefig(f"dataset_health/preparation/balancing_images/covid_over_eval.png")

figure()
eval: dict[str, list] = evaluate_approach(df_under, test_copy2, target=target, metric="recall", estimators_names= ["GaussianNB", "BernoulliNB"])
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Covid under evaluation", percentage=True
)
savefig(f"dataset_health/preparation/balancing_images/covid_under_eval.png")

figure()
eval: dict[str, list] = evaluate_approach(df_smote, test, target=target, metric="recall", estimators_names= ["GaussianNB", "BernoulliNB"])
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"Covid SMOTE evaluation", percentage=True
)
savefig(f"dataset_health/preparation/balancing_images/covid_smote_eval.png")
