from config.dslabs_functions import read_train_test_from_files, CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart, naive_Bayes_study, plot_evaluation_results
from numpy import array, ndarray
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show

file_tag = "covid"
train_filename = "dataset_health/data/covid_under.csv"
test_filename = "dataset_health/data/test.csv"
target = "CovidPos"
eval_metric = "accuracy"


trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target
)



figure(figsize=(4,5))
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metric)
savefig(f"dataset_health/modeling/NB_images/{file_tag}_nb_{eval_metric}_study.png")



figure(figsize=(4,5))
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, "recall")
savefig(f"dataset_health/modeling/NB_images/{file_tag}_nb_recall_study.png")
show()



prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'dataset_health/modeling/NB_images/{file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
show()
