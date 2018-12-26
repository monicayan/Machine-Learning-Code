# Machine-Learning-Code
Coding assignments for Columbia University Course **COMS 4721 *Machine Learning for Data Science*** 



#### MNIST_NN

use nearest neighbor classifier to calssify hand-written numbers, and reporte training error and test error.



#### model_selection

tried *Logistic Regression, Multi-layer Perceptron Regressor*, and *Random Forest Regressor* with hyper-parameter tuning, and with final model achieve square loss risk below 0.45; conclude the conditional probability estimator.



#### equalized_odds

analysed the salary dataset and tried to predict whether a person makes more than $50,000 a year or not. Description of the dataset can be found at [here](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names) .

trained *Logistic Regression, Decision Tree, Random Forest* and *Bagging Classifier* with hyper-parameters tuning and achieved the final model. Report false positive and false negative rates in both `sex = make` and `sex = female` ; analyse the "equalized odds" problems on the results.



#### GD_LR

use gradient descent to optimize the logistic regression MLE parameters (with explicit affine expansion):

<a href="https://www.codecogs.com/eqnedit.php?latex=\min_{\beta_{0}&space;\in&space;\mathbb{R},&space;\beta{0}&space;\in&space;\mathbb{R}^d}&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;\{\ln&space;(1&plus;\exp(\beta_{0}&plus;x_i^\mathrm{T}\beta))&space;-&space;y_i(\beta_0&plus;x_i^{\mathrm{T}}\beta)\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_{\beta_{0}&space;\in&space;\mathbb{R},&space;\beta{0}&space;\in&space;\mathbb{R}^d}&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;\{\ln&space;(1&plus;\exp(\beta_{0}&plus;x_i^\mathrm{T}\beta))&space;-&space;y_i(\beta_0&plus;x_i^{\mathrm{T}}\beta)\}" title="\min_{\beta_{0} \in \mathbb{R}, \beta{0} \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^{n} \{\ln (1+\exp(\beta_{0}+x_i^\mathrm{T}\beta)) - y_i(\beta_0+x_i^{\mathrm{T}}\beta)\}" /></a>

and observe the convergence behavior; try different threhold for convergence.



#### CA_soft-margin_SVM

implement coordinate ascent algorithm for soft-margin kernel SVM.



#### Restaurant Review Classification

use unigram, bigram, trigram and tf-idf as features, online-preceptron as classifier to predict whether a review is over 4-star or not.







