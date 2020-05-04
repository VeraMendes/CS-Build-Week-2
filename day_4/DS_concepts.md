# DS concepts

## Questions and Answers

### Statistics and Modeling

What is the Central Limit Theorem and why is it important?

What is sampling? How many sampling methods do you know?

What is the difference between type I vs type II error?

What is linear regression? What do the terms p-value, coefficient, and r-squared value mean? What is the significance of each of these components?

What are the assumptions required for linear regression?

What is a statistical interaction?

What is selection bias?

What is an example of a data set with a non-Gaussian distribution?

What is the Binomial Probability Formula?

What's the normal distribution? Why do we care about it?

How do we check if a variable follows the normal distribution?

What is the sparsity problem particular to random forests & boosting models?

Suppose you have 1000 features and n=10million. What is a suitable number of trees to start looking for in a random forest?

How is feature importance calculated for random forest, XGBoost, etc?

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

When should you consider adding a multiple testing penalty? What are the tradeoffs for each of the different types of penalties.

What is PCA? When do we need to do PCA? How do you select the right number of components?

What is stacking? Give an example.

Suppose you have n=200, and 10 features. Should you be conservative in your hyperparameter space if you will not be using Bayesian Optimization? Why or why not? Can it lead to overfitting or underfitting?

Suppose you have continuous data for heights. In addition, you have categorical variables "small height", "med height", "tall height" which was manually created by a former data scientist. Should you include all features? Are the categorical versions necessary in random forest or boosting models?

How would you treat categorical features?

What kinds of imputation techniques would you use for numeric and categorical features? What are the pros and cons to each?

Do you need a separate validation set if you will be using cross-fold validation?

When should you touch the test set? What is the maximum number of times you can touch the test set?

What is supervised machine learning?

What is regression? Which models can you use to solve a regression problem?

What is linear regression? When do we use it?

What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices?

What are the methods for solving linear regression do you know?

What is gradient descent? How does it work?

What is the normal equation?

What is SGD - stochastic gradient descent? What's the difference with the usual gradient descent?

Which metrics for evaluating regression models do you know?

What are MSE and RMSE?

What is overfitting?

How to do you validate your models?

Why do we need to split our data into three parts: train, validation, and test?

Can you explain how cross-validation works?

What is K-fold cross-validation?

How do we choose K in K-fold cross-validation? What's your favourite K?

What happens to our linear regression model if we have three columns in our data: x, y, z - and z is a sum of x and y?

What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise?

What is regularization? Why do we need it?

Which regularization techniques do you know?

What is classification? Which models would you use to solve a classification problem?

What is logistic regression? When do we need to use it?

Is logistic regression a linear model? Why?

What is sigmoid? What does it do?

How do we evaluate classification models?

What is accuracy?

Is accuracy always a good metric?

What is the confusion table? What are the cells in this table?

What is precision, recall, and F1-score?

Precision-recall trade-off

What is the ROC curve? When to use it?

What is AUC (AU ROC)? When to use it?

How to interpret the AU ROC score?

What is the PR (precision-recall) curve?

What is the area under the PR curve? Is it a useful metric?

In which cases AU PR is better than AU ROC?

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

Which feature selection techniques do you know?

Can we use L1 regularization for feature selection?

Can we use L2 regularization for feature selection?

What are the decision trees?

How do we train decision trees?

What are the main parameters of the decision tree model?

How do we handle categorical variables in decision trees?

What are the benefits of a single decision tree compared to more complex models?

How can we know which features are more important for the decision tree model?

What is random forest?

Why do we need randomization in random forest?

What are the main parameters of the random forest model?

How do we select the depth of the trees in random forest?

How do we know how many trees we need in random forest?

Is it easy to parallelize training of random forest? How can we do it?

What are the potential problems with many large trees?

What if instead of finding the best split, we randomly select a few splits and just select the best from them. Will it work?

What happens to random forest when we have correlated features in our data?

What is gradient boosting trees?

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

How does a usual fully-connected neural network work?

Why do we need activation functions?

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

What happens when the learning rate is too large? Too small?

How to choose the learning rate? 

What is Adam? What's the main difference between Adam and SGD?

When would you use Adam and when SGD?

Do we want to have a constant learning rate or we better change it throughout training?

How do we decide when to stop training a neural net?

What is model checkpointing?

Can you tell us how you approach the model training process?

How we can use neural nets for computer vision?

What is a convolution? What's a convolutional layer?

Why do we actually need convolutions? Can't we use a fully-connected layer for that?

What's pooling in CNN? Why do we need it?

How max pooling works? Are there other pooling techniques?

Are CNNs resistant to rotations? What happens to the predictions of a CNN if an image is rotated?

What are augmentations? Why do we need them?

What kind of augmentations do you know?

How to choose which augmentations to use?

What kind of CNN architectures for classification do you know?

What is transfer learning? How does it work?

What is object detection? Do you know any architectures for that?

What is object segmentation? Do you know any architectures for that?

How can we use machine learning for text classification?

What is bag of words? How we can use it for text classification?

What are the advantages and disadvantages of bag of words?

What are N-grams? How can we use them?

How large should be N for our bag of words when using N-grams?

What is TF-IDF? How it's useful for text classification?

Which model would you choose for text classification with bag of words features?

Would you prefer gradient boosting trees model or logistic regression when doing text classification with bag of words?

What are word embeddings? Why are they useful? Do you know Word2Vec?

Do you know any other ways to get word embeddings?

If you have a sentence with multiple words, you may need to combine multiple word embeddings into one. How would you do it?

Would you prefer gradient boosting trees model or logistic regression when doing text classification with embeddings?

How can you use neural nets for text classification?

How can we use CNN for text classification?

What is unsupervised learning?

What is clustering? When do we need it?

Do you know how K-means works?

How to select K for K-means?

What are the other clustering algorithms do you know?

Do you know how DBScan works?

When would you choose K-means and when DBScan?

What is the curse of dimensionality? Why do we care about it?

Do you know any dimensionality reduction techniques?

What's singular value decomposition? How is it typically used for machine learning? 

What is the ranking problem? Which models can you use to solve them?

What are good unsupervised baselines for text information retrieval?

How would you evaluate your ranking algorithms? Which offline metrics would you use?

What is precision and recall at k?

What is mean average precision at k?

How can we use machine learning for search?

How can we get training data for our ranking algorithms?

Can we formulate the search problem as a classification problem? How?

How can we use clicks data as the training data for ranking algorithms?

Do you know how to use gradient boosting trees for ranking?

How do you do an online evaluation of a new ranking algorithm?

What is a recommender system?

What are good baselines when building a recommender system?

What is collaborative filtering?

How we can incorporate implicit feedback (clicks, etc) into our recommender systems?

What is the cold start problem?

Possible approaches to solving the cold start problem?

What is a time series? 

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

How would you create a logistic regression model?

Have you used a time series model? Do you understand cross-correlations with time lags?

Explain the 80/20 rule, and tell me about its importance in model validation.

Explain what precision and recall are. How do they relate to the ROC curve?

Explain the difference between L1 and L2 regularization methods.

What is root cause analysis?

What are hash table collisions?

What is an exact test?

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

Can you name four types of problems where it shines?

What is a labeled training set?

What are the two most common supervised tasks?

Can you name four common unsupervised tasks?

What type of ML algorithm would you use to allow a robot to walk in various unknown terrains?

What type of algorithm would you use to segment your customers into groups?

Would you frame the problem of spam detection as supervised or unsupervised?

What is an online learning system?

What is out-of-core learning?

What type of learning algorithm relies on a similarity measure to make predictions?

What is the difference between a model parameter and a learning algo’s hyperparameter?

What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?

Can you name four of the main challenges in ML?

If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?

What is a test set and why would you want to use it?

What is the purpose of a validation set?

What can go wrong if you tune hyperparameters using the test set?

What is repeated cross-validation and why would you prefer it to using a single validation set?

### Programming

With which programming languages and environments are you most comfortable working?

What are some pros and cons about your favorite statistical software?

Tell me about an original algorithm you’ve created.

Describe a data science project in which you worked with a substantial programming component. What did you learn from that experience?

Do you contribute to any open-source projects?

How would you clean a data set in (insert language here)?

Tell me about the coding you did during your last project?

What are two main components of the Hadoop framework?

Explain how MapReduce works as simply as possible.

How would you sort a large list of numbers?

Say you’re given a large data set. What would be your plan for dealing with outliers? How about missing values? How about transformations?

What modules/libraries are you most familiar with? What do you like or dislike about them?

In Python, how is memory managed?

What are the supported data types in Python?

What is the difference between a tuple and a list in Python?

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

### Culture Fit

Which data scientists do you admire most? Which startups?

What do you think makes a good data scientist?

How did you become interested in data science?

Give a few examples of “best practices” in data science.

What is the latest data science book / article you read? What is the latest data mining conference / webinar / class / workshop / training you attended?

What’s a project you would want to work on at our company?

What unique skills do you think you’d bring to the team?

What data would you love to acquire if there were no limitations?

Have you ever thought about creating your own startup? Around which idea / concept?

What can your hobbies tell me that your resume can’t?

What are your top 5 predictions for the next 20 years?

What did you do today? Or what did you do this week / last week?

If you won a million dollars in the lottery, what would you do with the money?

What is one thing you believe that most people do not?

What personality traits do you butt heads with?

What (outside of data science) are you passionate about?

### Past experience (behavior)

Tell me about a time when you took initiative.

Tell me about a time when you had to overcome a dilemma.

Tell me about a time when you resolved a conflict.

Tell me about a time you failed and what you have learned from it.

Tell me about (a job on your resume). Why did you choose to do it and what do you like most about it?

Tell me about a challenge you have overcome while working on a group project.

When you encountered a tedious, boring task, how would you deal with it and motivate yourself to complete it?

What have you done in the past to make a client satisfied/happy?

What have you done in your previous job that you are really proud of?

What do you do when your personal life is running over into your work life?
