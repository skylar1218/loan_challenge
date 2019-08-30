#https://shirinsplayground.netlify.com/2018/11/ml_basics_gbm/
#https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
#http://proceedings.mlr.press/v42/chen14.pdf

#set directory
setwd("C:/Users/QUANH3/OneDrive - The Toronto-Dominion Bank/Desktop")

#install all package
install.packages("readr")
install.packages("plyr")
install.packages("lubridate")
install.packages("dplyr")
install.packages("tidyverse")
install.packages("fastDummies")
install.packages("chron")
install.packages("xgboost")
install.packages("caTools")
install.packages("e1071")
install.packages("caret")
install.packages("mlr")

library(readr)
library(plyr)
library(dplyr)
library(tidyverse)
library(chron)
library(e1071)

#load CSV file; change the file name accordingly
data <- read.csv('LoanStats_2018Q1_ds.csv')

#check the class of data frame
class(data)

#check dim of data
dim(data)

#check summary of all variables
summary(data)

#save summary output
sink("Loan Data Summary.txt")
print(summary(data))
sink() 

#number of missing/% of missing value
colMeans(is.na(data))

#histogram/box plot
hist(data$num_tl_30dpd)
boxplot(data$emp_length)

#boxplot for categorical variable
barplot(table(data$sub_grade))

#unique value for categorical variable
with(data,table(num_tl_30dpd))
class(data$loan_amnt)

#DATA CLEANING
#1. create variable
mondf <- function(d1, d2) {
  if (!requireNamespace("lubridate", quietly = TRUE)) {
    stop("lubridate needed for this function to work. Please install it.",
         call. = FALSE)
  }
  library(lubridate)
  
  monnb <- function(d) { lt <- as.POSIXlt(as.Date(d, origin="1900-01-01")); lt$year*12 + lt$mon }
  monnb(d2) - monnb(d1)
}

data$age_of_oldest_cr_line_in_mth <- mondf(as.Date(chron(format(as.Date(paste("01-", data$earliest_cr_line , sep =""), "%d-%b-%y"), "%m/%d/%y"))), '2019-08-27')
data$bad <- ifelse(data$loan_status %in% c("Charged Off", "Default", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"), 1, 0)

#drop '%' for interest rate
data$int_rate <- as.numeric(sub("%", "", data$int_rate))
data$revol_util <- as.numeric(sub("%", "", data$revol_util))

#trim leading whitespaces
data$term <- trimws(data$term, "l")

#2. create dummy
library(fastDummies)

data_w_dummy <- dummy_cols(data,
                           select_columns = c("term", "grade", "sub_grade", "home_ownership", "verification_status", "initial_list_status", "application_type", "debt_settlement_flag"),
                           remove_first_dummy = TRUE)

#3. check variable with more than 70% missing value
var_w_70pct_missing <- data[, which(colMeans(is.na(data)) > 0.7)]
names(var_w_70pct_missing)
colMeans(is.na(var_w_70pct_missing)) 

#4. drop variable
data_clean = subset(data_w_dummy, select = -c(issue_d, earliest_cr_line, term, sub_grade, grade, home_ownership, verification_status, initial_list_status, application_type, debt_settlement_flag, id, member_id, emp_title, emp_length, loan_status, last_pymnt_d, next_pymnt_d, last_credit_pull_d, annual_inc_joint, sec_app_earliest_cr_line, sec_app_inq_last_6mths, sec_app_mort_acc, sec_app_open_acc, sec_app_revol_util, sec_app_open_act_il, sec_app_num_rev_accts, sec_app_chargeoff_within_12_mths, sec_app_collections_12_mths_ex_med, sec_app_mths_since_last_major_derog, policy_code, dti_joint, verification_status_joint))

colnames(data_clean)[which(names(data_clean) == "term_36 months")] <- "term_36_months"
colnames(data_clean)[which(names(data_clean) == "verification_status_Not Verified")] <- "verification_status_Not_Verified"
colnames(data_clean)[which(names(data_clean) == "application_type_Joint App")] <- "application_type_Joint_App"

#5. recode missing value
data_clean[is.na(data_clean)] <- -1
summary(data_clean)

#bad rate
sum(data_clean$bad)/count(data_clean)
#0.08894

#-----------------------------------------------#
data_test <- select(data_clean, mths_since_last_delinq, mths_since_last_major_derog, mths_since_recent_bc_dlq, mths_since_recent_revol_delinq, bad)

#check mths_since_last_delinq
data_test2 <- aggregate(data_test[, 5], list(data_test$mths_since_last_delinq), mean)
data_test2
plot(data_test2$Group.1,data_test2$x,ylim=c(0,1.2),xlab="mths_since_last_delinq",ylab="bad rate")
#Group.1          x
#     -1 0.08469781
#      0 1.00000000
#      1 0.09090909
#      2 0.10909091
#      3 0.13333333
#      4 0.11023622
#      5 0.10676157
#      6 0.11384615
#      7 0.10736196
#      8 0.08783784
#      9 0.07920792
#     10 0.10000000
#     11 0.11363636
#     12 0.11793612

#check mths_since_last_major_derog
data_test2 <- aggregate(data_test[, 5], list(data_test$mths_since_last_major_derog), mean)
data_test2
plot(data_test2$Group.1,data_test2$x,xlab="mths_since_last_major_derog",ylab="bad rate")
#Group.1          x
#     -1 0.08579958
#      0 0.00000000
#      1 0.25000000
#      2 0.00000000
#      3 0.08333333
#      4 0.10204082
#      5 0.11764706
#      6 0.07246377
#      7 0.10769231
#      8 0.14084507
#      9 0.05194805
#     10 0.16455696
#     11 0.12765957
#     12 0.12745098

#check mths_since_recent_bc_dlq
data_test2 <- aggregate(data_test[, 5], list(data_test$mths_since_recent_bc_dlq), mean)
data_test2
plot(data_test2$Group.1,data_test2$x,xlab="mths_since_recent_bc_dlq",ylab="bad rate")
#Group.1          x
#     -1 0.08711886
#      0 1.00000000
#      1 0.10000000
#      2 0.03333333
#      3 0.15686275
#      4 0.12345679
#      5 0.11111111
#      6 0.16417910
#      7 0.15662651
#      8 0.06315789
#      9 0.10476190
#     10 0.10714286
#     11 0.09909910
#     12 0.11111111

#check mths_since_recent_revol_delinq
data_test2 <- aggregate(data_test[, 5], list(data_test$mths_since_recent_revol_delinq), mean)
data_test2
plot(data_test2$Group.1,data_test2$x,xlab="mths_since_recent_revol_delinq",ylab="bad rate")
#Group.1          x
#     -1 0.08704403
#      0 1.00000000
#      1 0.08571429
#      2 0.05000000
#      3 0.15596330
#      4 0.10948905
#      5 0.10738255
#      6 0.13068182
#      7 0.08720930
#      8 0.06707317
#      9 0.09782609
#     10 0.09027778
#     11 0.13089005
#     12 0.10727969
#-----------------------------------------------#

#split into train and test dataset
library(caTools)

set.seed(1234)
split = sample.split(data_clean$bad, SplitRatio = 0.7)
training_set = subset(data_clean, split == TRUE)
test_set = subset(data_clean, split == FALSE)

sum(training_set$bad)/count(training_set)
#0.08894286
sum(test_set$bad)/count(test_set)
#0.08893333

library(xgboost)

#preparing matrix
myvars <- names(training_set) %in% c("bad")
dtrain <- xgb.DMatrix(as.matrix(training_set[!myvars]), 
                      label = as.numeric(training_set$bad))
dtest <- xgb.DMatrix(as.matrix(test_set[!myvars]), 
                     label = as.numeric(test_set$bad))

#General Parameters: Controls the booster type in the model which eventually drives overall functioning
#Booster Parameters: Controls the performance of the selected booster
#Learning Task Parameters: Sets and evaluates the learning process of the booster from the given data

#General Parameters
#1. Booster[default=gbtree]
#Sets the booster type (gbtree, gblinear or dart) to use.
#For classification problems, you can use gbtree, dart. For regression, you can use any.

#2. nthread[default=maximum cores available]
#Activates parallel computation. 
#Generally, people don't change it as using maximum cores leads to the fastest computation.

#3. silent[default=0]
#If you set it to 1, your R console will get flooded with running messages. 
#Better not to change it.

#Booster Parameters
#1. nrounds[default=100]
#It controls the maximum number of iterations. 
#For classification, it is similar to the number of trees to grow.
#Should be tuned using CV

#2. eta[default=0.3][range: (0,1)]
#It controls the learning rate, i.e., the rate at which our model learns patterns in data. 
#After every round, it shrinks the feature weights to reach the best optimum.
#Lower eta leads to slower computation. It must be supported by increase in nrounds.
#Typically, it lies between 0.01 - 0.3

#3. gamma[default=0][range: (0,Inf)]
#It controls regularization (or prevents overfitting). 
#The optimal value of gamma depends on the data set and other parameter values.
#Higher the value, higher the regularization. 
#Regularization means penalizing large coefficients which don't improve the model's performance. 
#default = 0 means no regularization.
#Tune trick: Start with 0 and check CV error rate. 
#If you see train error >>> test error, bring gamma into action. 
#Higher the gamma, lower the difference in train and test CV. 
#If you have no clue what value to use, use gamma=5 and see the performance. 
#Remember that gamma brings improvement when you want to use shallow (low max_depth) trees.

#4. max_depth[default=6][range: (0,Inf)]
#It controls the depth of the tree.
#Larger the depth, more complex the model; higher chances of overfitting. 
#There is no standard value for max_depth. 
#Larger data sets require deep trees to learn the rules from data.
#Should be tuned using CV

#5. min_child_weight[default=1][range:(0,Inf)]
#In regression, it refers to the minimum number of instances required in a child node. In classification, if the leaf node has a minimum sum of instance weight (calculated by second order partial derivative) lower than min_child_weight, the tree splitting stops.
#In simple words, it blocks the potential feature interactions to prevent overfitting. Should be tuned using CV.

#6. subsample[default=1][range: (0,1)]
#It controls the number of samples (observations) supplied to a tree.
#Typically, its values lie between (0.5-0.8)

#7. colsample_bytree[default=1][range: (0,1)]
#It control the number of features (variables) supplied to a tree
#Typically, its values lie between (0.5,0.9)

#8. lambda[default=0]
#It controls L2 regularization (equivalent to Ridge regression) on weights. It is used to avoid overfitting.

#9. alpha[default=1]
#It controls L1 regularization (equivalent to Lasso regression) on weights. In addition to shrinkage, enabling alpha also results in feature selection. Hence, it's more useful on high dimensional data sets.

#Learning Task Parameters
#1. Objective[default=reg:linear]
#reg:linear - for linear regression
#binary:logistic - logistic regression for binary classification. It returns class probabilities
#multi:softmax - multiclassification using softmax objective. It returns predicted class labels. It requires setting num_class parameter denoting number of unique prediction classes.
#multi:softprob - multiclassification using softmax objective. It returns predicted class probabilities.

#2. eval_metric [no default, depends on objective selected]
#These metrics are used to evaluate a model's accuracy on validation data.
#For regression, default metric is RMSE. For classification, default metric is error.
#Available error functions are as follows:
##mae - Mean Absolute Error (used in regression)
##Logloss - Negative loglikelihood (used in classification)
##AUC - Area under curve (used in classification)
##RMSE - Root mean square error (used in regression)
##error - Binary classification error rate [#wrong cases/#all cases]
##mlogloss - multiclass logloss (used in classification)

#default parameters
params <- list(booster = "gbtree",
                 objective = "binary:logistic",
                 eta = 0.3, 
                 gamma = 0, 
                 max_depth = 6, 
                 min_child_weight = 1, 
                 subsample = 1, 
                 colsample_bytree = 1
               )

#define a watchlist for evaluating model performance during the training run
watchlist <- list(train = dtrain, eval = dtest)

#use xgb.cv function to calculate the best nround 
#also, this function also returns CV error, which is an estimate of test error.
xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                nrounds = 100, 
                nfold = 5, 
                showsd = T, 
                stratified = T, 
                print.every.n = 10, 
                early.stop.round = 20, 
                maximize = F
                )
##Best iteration:
##[63]	train-error:0.003214+0.000369	test-error:0.017229+0.001208

min(xgbcv$evaluation_log[, test_error_mean])
##0.0172286
##CV accuracy (100 - 1.72) = 98.28%

#calculate test set accuracy and determine if this default model makes sense
#first default - model training
xgb1 <- xgb.train(params = params,
                  data = dtrain,
                  nrounds = 63,
                  watchlist = watchlist, 
                  print.every.n = 10, 
                  early.stop.round = 10, 
                  maximize = F , 
                  eval_metric = "error"
                  )

#model prediction
xgbpred <- predict (xgb1, dtest)
xgbpred <- ifelse (xgbpred > 0.5, 1, 0) #convert output probabilities to labels; use cutoff = 0.5

#calculate model's accuracy using confusionMatrix() function from caret package
#confusion matrix
library(caret)

ts_label <- as.numeric(test_set$bad)
confusionMatrix(
  factor(xgbpred, levels = min(ts_label):max(ts_label)),
  factor(ts_label, levels = min(ts_label):max(ts_label))
  )
##Accuracy : 0.9827

#view variable importance plot
mat <- xgb.importance(feature_names = colnames(training_set), model = xgb1)
xgb.ggplot.importance(importance_matrix = mat[1:20]) 

#parameter tuning
#random/grid search procedure and find better accuracy
library(mlr)

#convert "bad" to factor
training_set[,'bad']<-factor(training_set[,'bad'])
test_set[,'bad']<-factor(test_set[,'bad'])

traintask <- makeClassifTask(data = training_set, target = "bad")
testtask <- makeClassifTask (data = test_set,target = "bad")

#create learner
lrn <- makeLearner("classif.xgboost",
                   predict.type = "response"
                   )

lrn$par.vals <- list(objective = "binary:logistic", 
                     eval_metric = "error", 
                     nrounds = 63L,
                     eta = 0.3
                     )

#set parameter space
params <- makeParamSet(makeDiscreteParam("booster", values = c("gbtree", "gblinear")), 
                       makeIntegerParam("max_depth", lower = 3L, upper = 20L), 
                       makeNumericParam("min_child_weight", lower = 1L, upper = 10L), 
                       makeNumericParam("subsample", lower = 0.5, upper = 1),
                       makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
                       makeNumericParam("gamma", lower = 0, upper = 20)
                       )

#set resampling strategy
rdesc <- makeResampleDesc("CV", stratify = T, iters=5L)
#with stratify=T, we'll ensure that distribution of target class is maintained in the resampled data sets

#set search optimization strategy
#use random search to find the best parameters
#in random search, we'll build 100 models with different parameters
#and choose the one with the least error
ctrl <- makeTuneControlRandom(maxit = 10L)

#set a parallel backend to ensure faster computation
library(parallel)
library(parallelMap) 

parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, 
                     task = traintask, 
                     resampling = rdesc, 
                     measures = acc, 
                     par.set = params, 
                     control = ctrl, 
                     show.info = T
                     )
##[Tune] Result: booster=gbtree; max_depth=20; min_child_weight=6.27; subsample=0.837; colsample_bytree=0.514; gamma=11.4 : acc.test.mean=0.9834571

mytune$y 
##acc.test.mean 
##0.9834571

#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune, task = traintask)

#predict model
xgpred <- predict(xgmodel, testtask)

confusionMatrix(xgpred$data$response, xgpred$data$truth)
##Accuracy : 0.983