import sys
sys.path.append('.')
sys.path.append('config')

from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig, show
from sklearn.linear_model import LogisticRegression
from dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from dslabs_functions import plot_evaluation_results, plot_multiline_chart, logistic_regression_study


file_tag = "covid"
file_dir = "dataset_health/modeling/LogisticRegression_images/"
train_filename = "dataset_health/data/covid_under.csv"
test_filename = "dataset_health/data/test.csv"
target = "CovidPos"
eval_metric = "accuracy" 


trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target
)


print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

figure()
best_model, params = logistic_regression_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_iterations=5000,
    lag=500,
    metric=eval_metric,
)
savefig(f"dataset_health/modeling/LogisticRegression_images/{file_tag}_lr_{eval_metric}_study.png")
# show()


prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)

figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f"{file_dir}{file_tag}_lr_{params['name']}_best_{params['metric']}_eval.png")
# show()


type: str = params["params"][0]
nr_iterations: list[int] = [i for i in range(100, 1001, 100)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric = "accuracy"

warm_start = False
for n in nr_iterations:
    clf = LogisticRegression(
        warm_start=warm_start,
        penalty=type,
        max_iter=n,
        solver="liblinear",
        verbose=False,
    )
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))
    warm_start = True

figure()
plot_multiline_chart(
    nr_iterations,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"LR overfitting study for penalty={type}",
    xlabel="nr_iterations",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"{file_dir}{file_tag}_lr_{type}_overfitting.png")