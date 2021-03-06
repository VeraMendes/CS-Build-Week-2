# DS concepts

## Questions and Answers


### Statistics and Modeling

What is the Central Limit Theorem and why is it important?  
The central limit theorem tells us exactly what the shape of the distribution of means will be when we draw repeated samples from a given population. Specifically, as the sample sizes get larger, the distribution of means calculated from repeated sampling will approach normality.
The central limit theorem tells us that no matter what the distribution of the population is, the shape of the sampling distribution will approach normality as the sample size (N) increases.

What is sampling? How many sampling methods do you know?  
Sampling methods are the ways to choose people from the population to be considered in a sample survey. Samples can be divided based on following criteria. Probability samples - In such samples, each population element has a known probability or chance of being chosen for the sample.
Random, Systematic, Convenience, Cluster, and Stratified. Random sampling is analogous to putting everyone's name into a hat and drawing out several names. Each element in the population has an equal chance of occuring.
Random sampling is analogous to putting everyone's name into a hat and drawing out several names. Each element in the population has an equal chance of occuring. While this is the preferred way of sampling, it is often difficult to do. It requires that a complete list of every element in the population be obtained. Computer generated lists are often used with random sampling. You can generate random numbers using the TI82 calculator.
Systematic sampling is easier to do than random sampling. In systematic sampling, the list of elements is "counted off". That is, every kth element is taken. This is similar to lining everyone up and numbering off "1,2,3,4; 1,2,3,4; etc". When done numbering, all people numbered 4 would be used.
Convenience sampling is very easy to do, but it's probably the worst technique to use. In convenience sampling, readily available data is used. That is, the first people the surveyor runs into.
Cluster sampling is accomplished by dividing the population into groups -- usually geographically. These groups are called clusters or blocks. The clusters are randomly selected, and each element in the selected clusters are used.
Stratified sampling also divides the population into groups called strata. However, this time it is by some characteristic, not geographically. For instance, the population might be separated into males and females. A sample is taken from each of these strata using either random, systematic, or convenience sampling.

What is the difference between type I vs type II error?  
In statistical hypothesis testing, a type I error is the rejection of a true null hypothesis, while a type II error is the non-rejection of a false null hypothesis.

What is the elbow method and why do we use it?    
n cluster analysis, the elbow method is a heuristic used in determining the number of clusters in a data set. The method consists of plotting the explained variation as a function of the number of clusters, and picking the elbow of the curve as the number of clusters to use.

What is linear regression? What do the terms p-value, coefficient, and r-squared value mean? What is the significance of each of these components?  
In statistics, linear regression is a linear approach to modeling the relationship between a scalar response and one or more explanatory variables. The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.
p-values and R-squared values measure different things. The p-value indicates if there is a significant relationship described by the model, and the R-squared measures the degree to which the data is explained by the model.
P-values and coefficients in regression analysis work together to tell you which relationships in your model are statistically significant and the nature of those relationships. The coefficients describe the mathematical relationship between each independent variable and the dependent variable. The p-values for the coefficients indicate whether these relationships are statistically significant.


What are the assumptions required for linear regression?  
There are four assumptions associated with a linear regression model:
Linearity: The relationship between X and the mean of Y is linear.
Homoscedasticity: The variance of residual is the same for any value of X.
Independence: Observations are independent of each other.
Normality: For any fixed value of X, Y is normally distributed.

What is a statistical interaction?  
In statistics, an interaction may arise when considering the relationship among three or more variables, and describes a situation in which the effect of one causal variable on an outcome depends on the state of a second causal variable.

What is selection bias?  
selection bias is the bias introduced by the selection of individuals, groups or data for analysis in such a way that proper randomization is not achieved, thereby ensuring that the sample obtained is not representative of the population intended to be analyzed. It is sometimes referred to as the selection effect.

What is an example of a data set with a non-Gaussian distribution?  
Any distribution of money or value will be non--Gaussian. For example: distributions of income; distributions of house prices; distributions of bets placed on a sporting event. These distributions cannot have negative values and will usually have extended right hand tails.

What is the Binomial Probability Formula?  
For the coin flip example, N = 2 and π = 0.5. The formula for the binomial distribution is shown below: where P(x) is the probability of x successes out of N trials, N is the number of trials, and π is the probability of success on a given trial.

What's the normal distribution? Why do we care about it?  
The Data Behind the Bell Curve
A normal distribution of data is one in which the majority of data points are relatively similar, meaning they occur within a small range of values with fewer outliers on the high and low ends of the data range.
Characteristics of Normal Distribution
Normal distributions are symmetric, unimodal, and asymptotic, and the mean, median, and mode are all equal. A normal distribution is perfectly symmetrical around its center. That is, the right side of the center is a mirror image of the left side.
In all normal or nearly normal distributions, there is a constant proportion of the area under the curve lying between the mean and any given distance from the mean when measured in standard deviation units. For instance, in all normal curves, 99.73 percent of all cases fall within three standard deviations from the mean, 95.45 percent of all cases fall within two standard deviations from the mean, and 68.27 percent of cases fall within one standard deviation from the mean.

How do we check if a variable follows the normal distribution?  
The Kolmogorov-Smirnov test (K-S) and Shapiro-Wilk (S-W) test are designed to test normality by comparing your data to a normal distribution with the same mean and standard deviation of your sample. If the test is NOT significant, then the data are normal, so any value above . 05 indicates normality.

What is the sparsity problem particular to random forests & boosting models?  


Suppose you have 1000 features and n=10million. What is a suitable number of trees to start looking for in a random forest?  


How is feature importance calculated for random forest, XGBoost, etc?  
Feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node. The node probability can be calculated by the number of samples that reach the node, divided by the total number of samples. The higher the value the more important the feature.

How do you determine important features from linear models?  


What happens to feature importance measures when there are highly related variables in a random forest/xgboost?


What are the most important hyperparameters in random forest, xgboost, etc.  


Suppose you have n = 20 and 200 features, is a random forest or xgboost model suitable?  


What is the loss function for SVM, random forest, linear regression, etc?  


Where/when is the regularization penalty applied i.e. the loss function or parameters?  


When is L1 not a suitable regularization penalty?  


Suppose you have n = 200. Is k=10 for cross-fold validation appropriate? If k=10 and test_size=0.3, what is the expected number of training samples per fold?  


What kinds of feature transformations are random forests and xgboost invariant to?  


How do you measure how well your clustering algorithm worked?  
Sum of within-cluster variance, W, is calculated for clustering analyses done with different values of k. W is a cumulative measure how good the points are clustered in the analysis. Plotting the k values and their corresponding sum of within-cluster variance helps in finding the number of clusters.

When should you consider adding a multiple testing penalty? What are the tradeoffs for each of the different types of penalties.  


What is PCA? When do we need to do PCA? How do you select the right number of components?  
Principal component analysis. PCA should be used mainly for variables which are strongly correlated. If the relationship is weak between variables, PCA does not work well to reduce data. Refer to the correlation matrix to determine. In general, if most of the correlation coefficients are smaller than 0.3, PCA will not help.Don’t choose the number of components manually. Instead of that, use the option that allows you to set the variance of the input that is supposed to be explained by the generated components. We know that the principal components explain a part of the variance. From the Scikit-learn implementation, we can get the information about the explained variance and plot the cumulative variance to check the number of principal components we need.

What is stacking? Give an example.  


Suppose you have n=200, and 10 features. Should you be conservative in your hyperparameter space if you will not be using Bayesian Optimization? Why or why not? Can it lead to overfitting or underfitting?  


Suppose you have continuous data for heights. In addition, you have categorical variables "small height", "med height", "tall height" which was manually created by a former data scientist. Should you include all features? Are the categorical versions necessary in random forest or boosting models?  


How would you treat categorical features?  
Numerical labels are always between 0 and n_classes-1. A common challenge with nominal categorical variable is that, it may decrease performance of a model.
Convert to Number
Using label encoder for conversion. ...
Create a new feature using mean or mode (most relevant value) of each age bucket.

What kinds of imputation techniques would you use for numeric and categorical features? What are the pros and cons to each?  


Do you need a separate validation set if you will be using cross-fold validation?  


When should you touch the test set? What is the maximum number of times you can touch the test set?  


What is supervised machine learning?  
Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples.

What is regression? Which models can you use to solve a regression problem?  


What is linear regression? When do we use it?  
Linear regression is the most basic and commonly used predictive analysis. Regression estimates are used to describe data and to explain the relationship.
It is the next step up after correlation. It is used when we want to predict the value of a variable based on the value of another variable. The variable we want to predict is called the dependent variable

What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices?  


What are the methods for solving linear regression do you know?  
The Linear Regression Equation
The equation has the form Y= a + bX, where Y is the dependent variable (that's the variable that goes on the Y axis), X is the independent variable (i.e. it is plotted on the X axis), b is the slope of the line and a is the y-intercept.

What is gradient descent? How does it work?  
Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model. With gradient descent we try to find the global minima, even though sometimes we get stuck on a local minima.

What is the normal equation?  
Normal equations are equations obtained by setting equal to zero the partial derivatives of the sum of squared errors (least squares); normal equations allow one to estimate the parameters of a multiple linear regression.

What is SGD - stochastic gradient descent? What's the difference with the usual gradient descent?  
So in a nutshell, Stochastic is one by one approach whereas, batch gradient descent is one go approach. Another difference between two is Stochastic gradient descent is faster than normal gradient descent.
Gradient descent is a simple optimization procedure that you can use with many machine learning algorithms. ... Stochastic gradient descent refers to calculating the derivative from each training data instance and calculating the update immediately.

Which metrics for evaluating regression models do you know?  
Mean Squared Error(MSE)
Root-Mean-Squared-Error(RMSE).
Mean-Absolute-Error(MAE).
R² or Coefficient of Determination.
Adjusted R²

What are MSE and RMSE?  
The Mean Squared Error (MSE) is a measure of how close a fitted line is to data points. ... The MSE has the units squared of whatever is plotted on the vertical axis. Another quantity that we calculate is the Root Mean Squared Error (RMSE). It is just the square root of the mean square error.
The smaller the Mean Squared Error, the closer the fit is to the data. The MSE has the units squared of whatever is plotted on the vertical axis. ... The RMSE is directly interpretable in terms of measurement units, and so is a better measure of goodness of fit than a correlation coefficient.

What is overfitting?  
In statistics, overfitting is "the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably". An overfitted model is a statistical model that contains more parameters than can be justified by the data. The essence of overfitting is to have unknowingly extracted some of the residual variation (i.e. the noise) as if that variation represented underlying model structure.

How to do you validate your models?  
The following methods for validation will be demonstrated: Train/test split.  
Splitting your data. The basis of all validation techniques is splitting your data when training your model.  
k-Fold Cross-Validation (k-Fold CV)  
Leave-one-out Cross-Validation (LOOCV)   
Nested Cross-Validation.  
Time Series CV.  
Comparing Models.

Why do we need to split our data into three parts: train, validation, and test?  


Can you explain how cross-validation works?  
Cross-validation is a technique used to protect against overfitting in a predictive model, particularly in a case where the amount of data may be limited. In cross-validation, you make a fixed number of folds (or partitions) of the data, run the analysis on each fold, and then average the overall error estimate.

What is K-fold cross-validation?  
k-Fold Cross-Validation. Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into.


How do we choose K in K-fold cross-validation? What's your favourite K?  


What happens to our linear regression model if we have three columns in our data: x, y, z - and z is a sum of x and y?  


What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise?  


What is regularization? Why do we need it?  
In mathematics, statistics, and computer science, particularly in machine learning and inverse problems, regularization is the process of adding information in order to solve an ill-posed problem or to prevent overfitting. Regularization applies to objective functions in ill-posed optimization problems. Regularization refers to the method of preventing overfitting, by explicitly controlling the model complexity. It leads to smoothening of the regression line and thus prevents overfitting. It does so by penalizing the bent of the regression line that tries to closely match the noisy data points.

Which regularization techniques do you know?  
Regularization techniques work by limiting the capacity of models—such as neural networks, linear regression, or logistic regression—by adding a parameter norm penalty Ω(θ) to the objective function. L2 regularization is also known as ridge regression or Tikhonov regularization.

What is classification? Which models would you use to solve a classification problem?  


What is logistic regression? When do we need to use it?  


Is logistic regression a linear model? Why?  
Logistic regression is considered a generalized linear model because the outcome always depends on the sum of the inputs and parameters. ... Logistic regression is an algorithm that learns a model for binary classification.

What is sigmoid? What does it do?  


How do we evaluate classification models?  
Via evaluation metrics for classification: accuracy, confusion matrix, ROC-AUC.

What is accuracy?  
the degree to which the result of a measurement, calculation, or specification conforms to the correct value or a standard.

Is accuracy always a good metric?
Not always, specially in case of imbalanced classes.

What is the confusion table? What are the cells in this table?  
In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning one.
Cells are: True Positives; False Positives; True Negatives; False Negatives.

What is precision, recall, and F1-score?  


Precision-recall trade-off  
Recall is the fraction of correct positives among the total positives in the dataset. ... This miss rate is either compromising precision or recall score. precision-recall tradeoff occur due to increasing one of the parameter(precision or recall) while keeping the model same.

What is the ROC curve? When to use it?
A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The ROC curve is created by plotting the true positive rate against the false positive rate at various threshold settings. ROC curves are frequently used to show in a graphical way the connection/trade-off between clinical sensitivity and specificity for every possible cut-off for a test or a combination of tests. In addition the area under the ROC curve gives an idea about the benefit of using the test(s) in question.

How to interpret the AU ROC score?
Accuracy is measured by the area under the ROC curve. An area of 1 represents a perfect test; an area of . 5 represents a worthless test.
...
The Area Under an ROC Curve
90-1 = excellent (A)
80-. 90 = good (B)
70-. 80 = fair (C)
60-. 70 = poor (D)
50-. 60 = fail (F)

What is the PR (precision-recall) curve?
A PR curve is simply a graph with Precision values on the y-axis and Recall values on the x-axis. ... It is important to note that Precision is also called the Positive Predictive Value (PPV). Recall is also called Sensitivity, Hit Rate or True Positive Rate (TPR)
A no-skill classifier is one that cannot discriminate between the classes and would predict a random class or a constant class in all cases.

What is the area under the PR curve? Is it a useful metric?  


In which cases AU PR is better than AU ROC?  
If one method is better in AU-ROC but worse in AU-PR, then the method is better in Recall but worse in Precision. So you should use this method when you want high recall. If one method is better in AU-PR but worse in AU-ROC, then the method is better in Precision but worse in Recall.

What do we do with categorical variables?  


Why do we need one-hot encoding?  


What kind of regularization techniques are applicable to linear models?  


How does L2 regularization look like in a linear model?  


How do we select the right regularization parameters?  


What's the effect of L2 regularization on the weights of a linear model?  


How L1 regularization looks like in a linear model?  


What's the difference between L2 and L1 regularization?  


Can we have both L1 and L2 regularization components in a linear model?  


What's the interpretation of the bias term in linear models?  


How do we interpret weights in linear models?  


If a weight for one variable is higher than for another - can we say that this variable is more important?  


When do we need to perform feature normalization for linear models? When it's okay not to do it?  


What is feature selection? Why do we need it?  


Is feature selection important for linear models?  
Feature Selection Techniques in Regression Model. ... Many times feature selection becomes very useful to overcome with overfitting problem. It helps us in determining the smallest set of features that are needed to predict the response variable with high accuracy.

Which feature selection techniques do you know?  


Can we use L1 regularization for feature selection?  


Can we use L2 regularization for feature selection?  


What are the decision trees?  
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.

How do we train decision trees?  


What are the main parameters of the decision tree model?  


How do we handle categorical variables in decision trees?  


What are the benefits of a single decision tree compared to more complex models?  


How can we know which features are more important for the decision tree model?  


What is random forest?  
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.

Why do we need randomization in random forest?  


What are the main parameters of the random forest model?  


How do we select the depth of the trees in random forest?  


How do we know how many trees we need in random forest?  


Is it easy to parallelize training of random forest? How can we do it?  


What are the potential problems with many large trees?  


What if instead of finding the best split, we randomly select a few splits and just select the best from them. Will it work?  


What happens to random forest when we have correlated features in our data?  


What is gradient boosting trees?  
Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. ... Explicit regression gradient boosting algorithms were subsequently developed by Jerome H.

What's the difference between random forest and gradient boosting?  


Is it possible to parallelize training of a gradient boosting model? How to do it?  


Feature importance in gradient boosting trees - what are possible options?  


Are there any differences between continuous and discrete variables when it comes to feature importance of gradient boosting models?  


What are the main parameters in the gradient boosting model?  


How do you approach tuning parameters in XGBoost or LightGBM?  


How do you select the number of trees in the gradient boosting model?  


Which parameter tuning strategies (in general) do you know?  


What's the difference between grid search parameter tuning strategy and random search? When to use one or another?  


What kind of problems neural nets can solve?  
Today, neural networks are used for solving many business problems such as sales forecasting, customer research, data validation, and risk management. For example, at Statsbot we apply neural networks for time-series predictions, anomaly detection in data, and natural language understanding.

How does a usual fully-connected neural network work?  


Why do we need activation functions?  
The purpose of an activation function is to add some kind of non-linear property to the function, which is a neural network. ... A neural network without any activation function would not be able to realize such complex mappings mathematically and would not be able to solve tasks we want the network to solve.

What are the problems with sigmoid as an activation function?  


What is ReLU? How is it better than sigmoid or tanh?  


How we can initialize the weights of a neural network?  


What if we set all the weights of a neural network to 0?  


What regularization techniques for neural nets do you know?  


What is dropout? Why is it useful? How does it work?  


What is backpropagation? How does it work? Why do we need it?  


Which optimization techniques for training neural nets do you know?  


How do we use SGD (stochastic gradient descent) for training a neural net?  


What's the learning rate?  
The amount that the weights are updated during training is referred to as the step size or the “learning rate.” Specifically, the learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value, often in the range between 0.0 and 1.0.

What happens when the learning rate is too large? Too small?  


How to choose the learning rate?  
There are multiple ways to select a good starting point for the learning rate. A naive approach is to try a few different values and see which one gives you the best loss without sacrificing speed of training. We might start with a large value like 0.1, then try exponentially lower values: 0.01, 0.001, etc.

What is Adam? What's the main difference between Adam and SGD?  
Adam is an adaptive learning rate optimization algorithm that's been designed specifically for training deep neural networks. ... The algorithms leverages the power of adaptive learning rates methods to find individual learning rates for each parameter.
First proposed in the 1950s, the technique can update each parameter of a model, observe how a change would affect the objective function, choose a direction that would lower the error rate, and continue iterating until the objective function converges to the minimum. SGD is a variant of gradient descent.

When would you use Adam and when SGD?  


Do we want to have a constant learning rate or we better change it throughout training?  


How do we decide when to stop training a neural net?  


What is model checkpointing?  
The TensorFlow save method saves three kinds of files because it stores the graph structure separately from the variable values. The . ... Even though there is no file named model. ckpt , you still refer to the saved checkpoint by that name when restoring it.

Can you tell us how you approach the model training process?  


How we can use neural nets for computer vision?  


What is a convolution? What's a convolutional layer?  
In mathematics convolution is a mathematical operation on two functions that produces a third function expressing how the shape of one is modified by the other. The term convolution refers to both the result function and to the process of computing it.
Convolutional layers are the major building blocks used in convolutional neural networks. A convolution is the simple application of a filter to an input that results in an activation. ... The result is highly specific features that can be detected anywhere on input images.

Why do we actually need convolutions? Can't we use a fully-connected layer for that?  


What's pooling in CNN? Why do we need it?  


How max pooling works? Are there other pooling techniques?  


Are CNNs resistant to rotations? What happens to the predictions of a CNN if an image is rotated?  


What are augmentations? Why do we need them?  
Data augmentation is a way of creating new 'data' with different orientations. The benefits of this are two fold, the first being the ability to generate 'more data' from limited data and secondly it prevents over fitting.
Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data.

What kind of augmentations do you know?  
Data augmentation techniques such as cropping, padding, and horizontal flipping are commonly used to train large neural networks.

How to choose which augmentations to use?  


What kind of CNN architectures for classification do you know?  
LeNet-5 architecture is perhaps the most widely known CNN architecture. It was created by Yann LeCun in 1998 and widely used for written digits recognition (MNIST). Here is the LeNet-5 architecture. We start off with a grayscale image (LeNet-5 was trained on grayscale images), with a shape of 32×32 x1.

What is transfer learning? How does it work?  
Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.
For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks.
We can use knowledge from one pretrained model to work with another dataset, using the neural network architecture and the weights for respective classes.

What is object detection? Do you know any architectures for that?  
Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class in digital images and videos. Well-researched domains of object detection include face detection and pedestrian detection.
R-CNN. R-CNN uses search selective method to find the regions to detect objects after it passes through convolutional networks. ...
Resnet50. The Resnet50 is a deep residual neural network that can also be used for object detection. ...
FPN. It uses pooling as a method of selection. ...
Retinanet. ...
Yolo V3/V2. ...
Faster R-CNN. ...
SSD.

What is object segmentation? Do you know any architectures for that?  


How can we use machine learning for text classification?  


What is bag of words? How we can use it for text classification?  
The bag-of-words model is a simplifying representation used in natural language processing and information retrieval. In this model, a text is represented as the bag of its words, disregarding grammar and even word order but keeping multiplicity. The bag-of-words model has also been used for computer vision. 
The bag-of-words model is commonly used in methods of document classification where the (frequency of) occurrence of each word is used as a feature for training a classifier. An early reference to "bag of words" in a linguistic context can be found in Zellig Harris's 1954 article on Distributional Structure.

What are the advantages and disadvantages of bag of words?  
Advantages:
Very simple to understand and implement.
Disadvantages:
Bag of words leads to a high dimensional feature vector due to large size of Vocabulary, V.
Bag of words doesn’t leverage co-occurrence statistics between words. In other words, it assumes all words are independent of each other. 
It leads to a highly sparse vectors as there is nonzero value in dimensions corresponding to words that occur in the sentence.

What are N-grams? How can we use them?  


How large should be N for our bag of words when using N-grams?  


What is TF-IDF? How it's useful for text classification?  
In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.
It is a technique which is used to find meaning of sentences consisting of words and cancels out the incapabilities of Bag of Words technique which is good for text classification or for helping a machine read words in numbers.


Which model would you choose for text classification with bag of words features?  


Would you prefer gradient boosting trees model or logistic regression when doing text classification with bag of words?  


What are word embeddings? Why are they useful? Do you know Word2Vec?  


Do you know any other ways to get word embeddings?  


If you have a sentence with multiple words, you may need to combine multiple word embeddings into one. How would you do it?  


Would you prefer gradient boosting trees model or logistic regression when doing text classification with embeddings?  


How can you use neural nets for text classification?  


How can we use CNN for text classification?  


What is unsupervised learning?  
Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels and with a minimum of human supervision.

What is clustering? When do we need it?  
Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups.
Clustering can be considered the most important unsupervised learning problem; so, as every other problem of this kind, it deals with finding a structure in a collection of unlabeled data. A loose definition of clustering could be “the process of organizing objects into groups whose members are similar in some way”.

Do you know how K-means works?  
The k-means clustering algorithm attempts to split a given anonymous data set (a set containing no information as to class identity) into a fixed number (k) of clusters. Initially k number of so called centroids are chosen. ... Each centroid is thereafter set to the arithmetic mean of the cluster it defines.

How to select K for K-means?  
The Elbow Method
This is probably the most well-known method for determining the optimal number of clusters. It is also a bit naive in its approach. Calculate the Within-Cluster-Sum of Squared Errors (WSS) for different values of k, and choose the k for which WSS becomes first starts to diminish.

What are the other clustering algorithms do you know?  


Do you know how DBScan works?  
DBSCAN works as such: Divides the dataset into n dimensions. ... DBSCAN counts this shape as a cluster. DBSCAN iteratively expands the cluster, by going through each individual point within the cluster, and counting the number of other data points nearby.

When would you choose K-means and when DBScan?  
 Density clustering algorithms use the concept of reachability i.e. how many neighbors has a point within a radius. DBScan is more lovely because it doesn't need parameter, k, which is the number of clusters we are trying to find, which KMeans needs.

What is the curse of dimensionality? Why do we care about it?  


Do you know any dimensionality reduction techniques?  
Seven Techniques for Data Dimensionality Reduction  
Missing Values Ratio. Data columns with too many missing values are unlikely to carry much useful information.  
Low Variance Filter.  
High Correlation Filter.  
Random Forests / Ensemble Trees.  
Principal Component Analysis (PCA).  
Backward Feature Elimination.  
Forward Feature Construction.  

What's singular value decomposition? How is it typically used for machine learning?   


What is the ranking problem? Which models can you use to solve them?  


What are good unsupervised baselines for text information retrieval?  


How would you evaluate your ranking algorithms? Which offline metrics would you use?  


What is precision and recall at k?  


What is mean average precision at k?  
Mean Average Precision at K. ... APK is a measure of the average relevance scores of a set of the top-K documents presented in response to a query. For each query instance, we will compare the set of top-K results with the set of actual relevant documents, that is, a ground truth set of relevant documents for the query.

How can we use machine learning for search?  


How can we get training data for our ranking algorithms?  


Can we formulate the search problem as a classification problem? How?  


How can we use clicks data as the training data for ranking algorithms?  


Do you know how to use gradient boosting trees for ranking?  


How do you do an online evaluation of a new ranking algorithm?  


What is a recommender system?  
A recommender system, or a recommendation system, is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. They are primarily used in commercial applications.

What are good baselines when building a recommender system?  


What is collaborative filtering?  
Collaborative filtering is a technique used by recommender systems. Collaborative filtering has two senses, a narrow one and a more general one.

How we can incorporate implicit feedback (clicks, etc) into our recommender systems?  
When the recommender system gives out some recommendations, we can check the ones that the user actually takes and the ones the user repeats and this will help to make the recommender system better based not only on the type of product but on the user preferences. We can make the preferences better for that user and using collaborative system, we are using a user's preferences to adapt them to other users with similar profiles.

What is the cold start problem?  
Cold start in computing refers to a problem where a system or its part was created or restarted and is not working at its normal operation. The problem can be related to initialising internal objects or populating cache or starting up subsystems.

Possible approaches to solving the cold start problem?  


What is a time series?   
A time series is a series of data points indexed in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data. 

How is time series different from the usual regression problem?  


Which models do you know for solving time series problems?  


If there's a trend in our series, how can we remove it? And why would we want to do it?  


You have a series with only one variable "y" measured at time t. How to predict "y" at time t+1? Which approaches would you use?  


You have a series with a variable "y" and a set of features. How do you predict "y" at t+1? Which approaches would you use?  


What are the problems with using trees for solving time series problems?   


Tell me about how you designed a model for a past employer or client.  


What are your favorite data visualization techniques?  


How would you effectively represent data with 5 dimensions?  


How is k-NN different from k-means clustering?  
KNN represents a supervised classification algorithm that will give new data points accordingly to the k number or the closest data points, while k-means clustering is an unsupervised clustering algorithm that gathers and groups data into k number of clusters.

How would you create a logistic regression model?  


Have you used a time series model? Do you understand cross-correlations with time lags?  


Explain the 80/20 rule, and tell me about its importance in model validation.  


Explain what precision and recall are. How do they relate to the ROC curve?  


Explain the difference between L1 and L2 regularization methods.  


What is root cause analysis?  
In science and engineering, root cause analysis is a method of problem solving used for identifying the root causes of faults or problems. It is widely used in IT operations, telecommunications, industrial process control, accident analysis, medicine, healthcare industry, etc.

What are hash table collisions?  
Hash table is a data structure that uses a hash function to map elements(keys) to an index. ... A collision occurs when two or more elements are hashed(mapped) to same value. For example: Let the hash function be hash(x) = x%10. In this case 25 and 35 would be mapped to same value. Hence a collision occurs.

What is an exact test?  
In statistics, an exact test is a test where if the null hypothesis is true then all assumptions, upon which the derivation of the distribution of the test statistic is based, are met.

In your opinion, which is more important when designing a machine learning model: model performance or model accuracy?  


What is one way that you would handle an imbalanced data set that’s being used for prediction (i.e., vastly more negative classes than positive classes)?  


How would you validate a model you created to generate a predictive model of a quantitative outcome variable using multiple regression?  


I have two models of comparable accuracy and computational performance. Which one should I choose for production and why?  


How do you deal with sparsity?  


Is it better to spend five days developing a 90-percent accurate solution or 10 days for 100-percent accuracy?  


What are some situations where a general linear model fails?  


Do you think 50 small decision trees are better than a large one? Why?  


When modifying an algorithm, how do you know that your changes are an improvement over not doing anything?  


Is it better to have too many false positives or too many false negatives?  


How would you define Machine Learning?  
Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.

Can you name four types of problems where it shines?  
Machine Learning is great for:
Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better.
Complex problems for which there is no good solution at all using a traditional approach: the best Machine Learning techniques can find a solution.
Fluctuating environments: a Machine Learning system can adapt to new data.
Getting insights about complex problems and large amounts of data.

What is a labeled training set?  
Labeled data is a group of samples that have been tagged with one or more labels. ... After obtaining a labeled dataset, machine learning models can be applied to the data so that new unlabeled data can be presented to the model and a likely label can be guessed or predicted for that piece of unlabeled data.

What are the two most common supervised tasks?  
The two most common supervised tasks are regression and classification.

Can you name four common unsupervised tasks?  


What type of ML algorithm would you use to allow a robot to walk in various unknown terrains?  


What type of algorithm would you use to segment your customers into groups?  


Would you frame the problem of spam detection as supervised or unsupervised?  


What is an online learning system?  
Online learning is education that takes place over the Internet. It is often referred to as “e- learning” among other terms. However, online learning is just one type of “distance learning” - the umbrella term for any learning that takes place across distance and not in a traditional classroom.

What is out-of-core learning?  
Out-of-core learning refers to a set of algorithms working with data that cannot fit into the memory of a single computer, but that can easily fit into some data storage such as a local hard disk or web repository.

What type of learning algorithm relies on a similarity measure to make predictions?  


What is the difference between a model parameter and a learning algo’s hyperparameter?  


What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?  


Can you name four of the main challenges in ML?  


If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?  


What is a test set and why would you want to use it?  


What is the purpose of a validation set?  
A validation set is a set of data used to train artificial intelligence (AI) with the goal of finding and optimizing the best model to solve a given problem. Validation sets are also known as dev sets.

What can go wrong if you tune hyperparameters using the test set?  


What is repeated cross-validation and why would you prefer it to using a single validation set?  


### Programming

With which programming languages and environments are you most comfortable working?
Python
Pipenv virtual environment, Anaconda Local environment, colab env

What are some pros and cons about your favorite statistical software?  


Tell me about an original algorithm you’ve created.  


Describe a data science project in which you worked with a substantial programming component. What did you learn from that experience?  


Do you contribute to any open-source projects?
All the projects I done up to now are open-source projects.

How would you clean a data set in (insert language here)?  


Tell me about the coding you did during your last project?  


What are two main components of the Hadoop framework?  
HDFS (storage) and MapReduce (processing) are the two core components of Apache Hadoop. 

Explain how MapReduce works as simply as possible.  
The core idea behind MapReduce is mapping your data set into a collection of <key, value> pairs, and then reducing over all pairs with the same key. The overall concept is simple, but is actually quite expressive when you consider that: ... Almost all data can be mapped into <key, value> pairs somehow, and 2.

How would you sort a large list of numbers?  


Say you’re given a large data set. What would be your plan for dealing with outliers? How about missing values? How about transformations?  


What modules/libraries are you most familiar with? What do you like or dislike about them?  


In Python, how is memory managed?
Memory management in Python involves a private heap containing all Python objects and data structures. The management of this private heap is ensured internally by the Python memory manager.

What are the supported data types in Python?  


What is the difference between a tuple and a list in Python?
 List and Tuple in Python are the class of data structure. The list is dynamic, whereas tuple has static characteristics. ... Tuple is also a sequence data type that can contain elements of different data types, but these are immutable in nature.

What are the different types of sorting algorithms available in R language?  


What are the different data objects in R?  


What packages are you most familiar with? What do you like or dislike about them?  


How do you access the element in the 2nd column and 4th row of a matrix named M?  


What is the command used to store R objects in a file?  


What is the best way to use Hadoop and R together for analysis?  


How do you split a continuous variable into different groups/ranks in R?  


Write a function in R language to replace the missing value in a vector with the mean of that vector.  


What is the purpose of the group functions in SQL? Give some examples of group functions.  


Tell me the difference between an inner join, left join/right join, and union.  


What does UNION do? What is the difference between UNION and UNION ALL?  


What is the difference between SQL and MySQL or SQL Server?  


If a table contains duplicate rows, does a query result display the duplicate values by default? How can you eliminate duplicate rows from a query result?  


### Problem-Solving

How would you come up with a solution to identify plagiarism?  


How many “useful” votes will a Yelp review receive?  


How do you detect individual paid accounts shared by multiple users?  


You are about to send a million emails. How do you optimize delivery? How do you optimize response?  


You have a data set containing 100,000 rows and 100 columns, with one of those columns being our dependent variable for a problem we’d like to solve. How can we quickly identify which columns will be helpful in predicting the dependent variable. Identify two techniques and explain them to me as though I were 5 years old.  


How would you detect bogus reviews, or bogus Facebook accounts used for bad purposes?
This is an opportunity to showcase your knowledge of machine learning algorithms; specifically, sentiment analysis and text analysis algorithms. Showcase your knowledge of fraudulent behavior—what are the abnormal behaviors that can typically be seen from fraudulent accounts?  


How would you perform clustering on a million unique keywords, assuming you have 10 million data points—each one consisting of two keywords, and a metric measuring how similar these two keywords are? How would you create this 10 million data points table in the first place?  


How would you optimize a web crawler to run much faster, extract better information, and better summarize data to produce cleaner databases?  


### Culture Fit

Which data scientists do you admire most? Which startups?  


What do you think makes a good data scientist?  
A curious person, someone who has a deep understanding of statistics and is able to process his/her analysis and modelling thoughts through code being able to present the results easily with good visualizations for a technical and non-technical audience. 

How did you become interested in data science?  


Give a few examples of “best practices” in data science.  


What is the latest data science book / article you read? What is the latest data mining conference / webinar / class / workshop / training you attended?  


What’s a project you would want to work on at our company?  
Data Analysis.

What unique skills do you think you’d bring to the team?  
My extensive healthcare experience and my previous management skills.

What data would you love to acquire if there were no limitations?  


Have you ever thought about creating your own startup? Around which idea / concept?  
I did not think about it, but if I would do it I believe it would be in Healthcare- Radiology.

What can your hobbies tell me that your resume can’t?  
Yoga - I can easily navigate situations of stress and I'm very careful with my health.
Swimming - I like complete sports as I like to deliver fully projects well finished paying attention to detail.
Hiking - I like to explore and find different solutions, so I can choose the best one for my problem.
Cooking - I am hwalthy in body and mind and I like my products to reflect that.

What are your top 5 predictions for the next 20 years?  


What did you do today? Or what did you do this week / last week?  


If you won a million dollars in the lottery, what would you do with the money?  
I would only work as an hobby, help my family and friends, invest my money and would like to travel the World. 

What is one thing you believe that most people do not?  


What personality traits do you butt heads with?  


What (outside of data science) are you passionate about?  
Yoga, travel and cultures, beach.

### Past experience (behavior)

Tell me about a time when you took initiative.  


Tell me about a time when you had to overcome a dilemma.  


Tell me about a time when you resolved a conflict.  
In the hospital, two of my staff members had a disagreement. I need them to work together, so I asked them to meet and solve their disagreement at least that in a work environment they would be able to be professional and work together. I told them if they would not solve the problem between them, I would have to intervein and meet with both of them to sort the matter. Happilly they were able to solve the situation between them two and I did not need to intervein. 

Tell me about a time you failed and what you have learned from it.  


Tell me about (a job on your resume). Why did you choose to do it and what do you like most about it?  
I love statistics. Data Science feels like being able to do magic, I love the outcomes of being able to use Data Science tools.

Tell me about a challenge you have overcome while working on a group project.  


When you encountered a tedious, boring task, how would you deal with it and motivate yourself to complete it?  
Create deadlines to myself. Find ways of making the task a little more interesting inside its area, maybe the team might help for that matter.

What have you done in the past to make a client satisfied/happy?  
I listen to them.

What have you done in your previous job that you are really proud of?  
I planned and executed the implementation of a new MRI system end-to-end which allowed the department to double its patient intake.

What do you do when your personal life is running over into your work life?  
I have to find a way of balancing it.


