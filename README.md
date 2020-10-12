#### Asssignment for Parallel Computing assignment
***
1. Prerequisite
    * Environment: Python 3, Pyspark
    * Cloud service account if you want to deploy on it.
2. Dataset
   * Task: Text classification
   * Name: Amazon review dataset
   * Description: Dataset contains a few million Amazon customer reviews in text and star rating which are output labels. Training set is about 1.6GB, test set is about 177MB.
   * Link: [Download](https://www.kaggle.com/bittlingmayer/amazonreviews). After downloading and extracting, we place it in *dataset* folder.
3. Works
   *  We use above dataset and combine with pyspark to evaluate the performance of Spark processing big data. We apply Machine learning supported by libraries of Spark to accomplish the task. The techniques we used here are TF-IDF for feature engineering step, and some different classification algorithms such as *gradient boost*, *logistic regression*, *support vector machine*, *random forest*. We can calculate time and accuracy per algorithm.
   *  Deployment
      * Now, we run the task on the local machine.
      * In the future, we will run on Super Node XP or AWS (charge).