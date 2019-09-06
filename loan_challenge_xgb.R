#https://shirinsplayground.netlify.com/2018/11/ml_basics_gbm/
#https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
#http://proceedings.mlr.press/v42/chen14.pdf

#install all package
install.packages("here")
install.packages("readr")
install.packages("data.table")
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
install.packages("DMwR")
install.packages("Ckmeans.1d.dp")
install.packages("xgboostExplainer")
install.packages("sets")
install.packages("pROC")
install.packages("ROCR")

library(here)
library(readr)
library(plyr)
library(dplyr)
library(tidyverse)
library(chron)
library(e1071)

#load CSV file
dataset_path <- here::here("LoanStats_2018Q1_ds.csv")
data <- read.csv(dataset_path)

#check overall variables summary
#summary(data)



#DATA CLEANING/PREPARATION
#create variable
#function mondf is used to calculate the # of months between dates, d1 and d2
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
data$pct_sats <- ifelse(data$total_acc == 0, 0, data$num_sats/data$total_acc)

#drop '%' for interest rate
data$int_rate <- as.numeric(sub("%", "", data$int_rate))
data$revol_util <- as.numeric(sub("%", "", data$revol_util))

#trim leading whitespaces
data$term <- trimws(data$term, "l")

library(sets)
char_col <- colnames(data)[sapply(data, is.character)]
for(i in char_col) set(data, j = i, value = str_trim(data[[i]],side = "left"))

#create dummy
library(fastDummies)

data_w_dummy <- dummy_cols(data,
                           select_columns = c("term", "grade", "sub_grade", "home_ownership", "verification_status", "initial_list_status", "application_type", "debt_settlement_flag"),
                           remove_first_dummy = TRUE)

#drop variable
data_clean = subset(data_w_dummy, select = -c(issue_d, earliest_cr_line, term, sub_grade, grade, home_ownership, verification_status, initial_list_status, application_type, debt_settlement_flag, id, member_id, emp_title, emp_length, loan_status, last_pymnt_d, next_pymnt_d, last_credit_pull_d, annual_inc_joint, sec_app_earliest_cr_line, sec_app_inq_last_6mths, sec_app_mort_acc, sec_app_open_acc, sec_app_revol_util, sec_app_open_act_il, sec_app_num_rev_accts, sec_app_chargeoff_within_12_mths, sec_app_collections_12_mths_ex_med, sec_app_mths_since_last_major_derog, policy_code, dti_joint, verification_status_joint))

colnames(data_clean)[which(names(data_clean) == "term_36 months")] <- "term_36_months"
colnames(data_clean)[which(names(data_clean) == "verification_status_Not Verified")] <- "verification_status_Not_Verified"
colnames(data_clean)[which(names(data_clean) == "application_type_Joint App")] <- "application_type_Joint_App"

#bad rate
prop.table(table(data_clean$bad))
##      0       1 
##0.91106 0.08894 

#split into train and test dataset
library(caTools)

set.seed(1234)
split = sample.split(data_clean$bad, SplitRatio = 0.7)
training_set = subset(data_clean, split == TRUE)
test_set = subset(data_clean, split == FALSE)

prop.table(table(training_set$bad))
##         0          1 
##0.91105714 0.08894286 
prop.table(table(test_set$bad))
##         0          1 
##0.91106667 0.08893333

#preparing matrix
library(xgboost)

myvars <- names(training_set) %in% c("bad")
dtrain <- xgb.DMatrix(as.matrix(training_set[!myvars]), 
                      label = as.numeric(training_set$bad))
dtest <- xgb.DMatrix(as.matrix(test_set[!myvars]), 
                     label = as.numeric(test_set$bad))



#XGB - HYPERPARAMETER TUNING (manually setting the parameters then repeat)
#run cross validation 20 time and each time with random parameters
#we get the best parameter set and the minimum test_error_mean in each iteration
#we then get the best global minimum test_error_mean and its index (round)
#test_error_mean is binary classification error rate 
#it is calculated as #(wrong cases)/#(all cases)
#for the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
best_param = list()
best_seednumber = 1234
best_error = Inf
best_error_index = 0
best_auc = 0

for (iter in 1:20) {
  param <- list(booster = "gbtree",
                objective = "binary:logistic",
                eval_metric = "error",
                eval_metric = "auc",
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, 1),
                colsample_bytree = runif(1, .5, 1), 
                min_child_weight = sample(1:40, 1)
  )
  cv.nround = 300
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data = dtrain, params = param, nthread = 6, 
                 nfold = cv.nfold, nrounds = cv.nround,
                 verbose = T, print_every_n = 50, early_stop_round = 8, maximize = FALSE)
  
  min_error = min(mdcv$evaluation_log[, test_error_mean])
  min_error_index = which.min(mdcv$evaluation_log[, test_error_mean])
  max_auc = max(mdcv$evaluation_log[, test_auc_mean])
  
  if (min_error < best_error) {
    best_error = min_error
    best_error_index = min_error_index
    best_seednumber = seed.number
    best_auc = max_auc
    best_param = param
  }
}

nround = best_error_index #best nround
set.seed(best_seednumber)
md <- xgb.train(data = dtrain,
                params = best_param, 
                nrounds = nround,
                nthread = 6)

#feature importance plot
mat1 <- xgb.importance(feature_names = colnames(training_set), model = md)
xgb.plot.importance(importance_matrix = mat1[0:20])
xgb.ggplot.importance(importance_matrix = mat1[0:20]) 
#dev.copy(png, 'mat1_feature_importance.png')
#dev.off()

#model prediction
xgbpred1_train <- predict(md, dtrain)
xgbpred1_test <- predict(md, dtest)

#ROC is being drawn and AUC is calculated sorting the prediction scores 
#and seeing what % of target events are found in the prediction set
#it is checking what % of target events you could find if you move the cutoff point
#train AUC value
library(pROC) 
tr_label <- as.numeric(training_set$bad)
par(pty = "s")
roc_training <- roc(tr_label,
                    xgbpred1_train,
                    algorithm = 2,
                    percent = T)
plot(roc_training, 
     print.auc = T, 
     col = "blue",
     print.auc.x = 40,
     print.auc.y = 95)
auc(roc_training)
##1

#test AUC value
ts_label <- as.numeric(test_set$bad)
roc_test <- roc(response = ts_label,
                predictor = xgbpred1_test,
                algorithm = 2, 
                percent = T) 
plot(roc_test,
     print.auc = T, 
     col = "red", 
     print.auc.x = 80, 
     print.auc.y = 95, 
     add = T)
#dev.copy(png, 'mat1_roc.png')
#dev.off()
auc(roc_test)
##0.9138

#ROC curve with cut-off points
library(ROCR)
xgb.pred <- prediction(xgbpred1_test, ts_label)
xgb.perf <- performance(xgb.pred, "tpr", "fpr")

plot(xgb.perf,
     avg = "threshold",
     colorize = TRUE,
     lwd = 1,
     main = "ROC Curve w/ Thresholds",
     print.cutoffs.at = seq(0, 1, by = 0.1),
     text.adj = c(-0.5, 0.5),
     text.cex = 0.5)

grid(col = "lightgray")
axis(1, at=seq(0, 1, by = 0.1))
axis(2, at=seq(0, 1, by = 0.1))
abline(v = c(0.1, 0.3, 0.5, 0.7, 0.9), col = "lightgray", lty = "dotted")
abline(h = c(0.1, 0.3, 0.5, 0.7, 0.9), col = "lightgray", lty = "dotted")
lines(x = c(0, 1), y = c(0, 1), col = "black", lty = "dotted")

#dev.copy(png, 'mat1_test_roc_with_cutoff.png')
#dev.off()

#convert output probabilities to labels
#use cutoff = 0.5
xgbpred1_train <- ifelse(xgbpred1_train > 0.5, 1, 0)
xgbpred1_test <- ifelse(xgbpred1_test > 0.5, 1, 0)

#calculate model's accuracy using confusionMatrix() function from caret package
library(caret)

confusionMatrix(
  factor(xgbpred1_test, levels = min(ts_label):max(ts_label)),
  factor(ts_label, levels = min(ts_label):max(ts_label))
)
##Accuracy : 0.9836



#XGB - RESULT VISUALIZATION
#xgboostExplainer
#https://medium.com/applied-data-science/new-r-package-the-xgboost-explainer-51dd7d1aa211
install.packages("remotes")
remotes::install_github("davidADSP/xgboostExplainer")
library(xgboostExplainer)

#create data table that stores the feature impact breakdown for each leaf of
#each tree in an xgboost model
explainer = buildExplainer(md,
                           dtrain, 
                           type = "binary", 
                           base_score = 0.5, 
                           trees_idx = NULL)

#create feature impact breakdown of a set of predictions made using an xgboost model
pred.breakdown = explainPredictions(md,  
                                    explainer,
                                    dtest)

cat('Breakdown Complete','\n')

#total weight/log-odds for each observation
weights = rowSums(pred.breakdown)

#convert log-odds to probability
pred.xgb = 1/(1+exp(-weights))

cat(max(xgbpred1_test-pred.xgb),'\n')

#pick a random observation
idx_to_get = as.integer(1765)

ix <- which(colnames(test_set) %in% c("bad")) 

showWaterfall(md, 
              explainer,
              dtest, 
              data.matrix(test_set[, -ix]),
              idx_to_get, 
              threshold = 0.4,
              type = "binary")

#dev.copy(png, 'md_logodds_waterfall.png')
#dev.off()

#impack against variable value plot
x1 <- data.frame(test_set[,"installment"], pred.breakdown[,"installment"])
plot(x1,
     cex = 0.4,
     pch = 16, 
     xlab = "Installment Level", 
     ylab = "Installment Level impact on log-odds")

#dev.copy(png, 'md_installment_level_impact_on_log-odds.png')
#dev.off()

x2 <- data.frame(test_set[,"total_rec_prncp"], pred.breakdown[,"total_rec_prncp"])
plot(x2,
     cex = 0.4,
     pch = 16, 
     xlab = "Total Principal Received To Date Level", 
     ylab = "Total Principal Received To Date Level impact on log-odds")

#dev.copy(png, 'md_total_rec_prncp_level_impact_on_log-odds.png')
#dev.off()


###############################################################################
# #XGB - HYPERPARAMETER TUNING (using mlr package)
# #use random/grid search procedure
# library(mlr)
# 
# #convert "bad" to factor
# training_set[,'bad']<-factor(training_set[,'bad'])
# test_set[,'bad']<-factor(test_set[,'bad'])
# 
# traintask <- makeClassifTask(data = training_set, target = "bad")
# testtask <- makeClassifTask (data = test_set,target = "bad")
# 
# #create learner
# lrn <- makeLearner("classif.xgboost",
#                    predict.type = "response"
# )
# 
# #use nround result from previous tuned nround
# lrn$par.vals <- list(objective = "binary:logistic",
#                      eval_metric = "auc",
#                      eval_metric = "error",
#                      nrounds = best_error_index
# )
# 
# #set parameter space
# params <- makeParamSet(makeDiscreteParam("booster", values = c("gbtree", "gblinear")),
#                        makeIntegerParam("max_depth", lower = 3L, upper = 20L),
#                        makeNumericParam("min_child_weight", lower = 1L, upper = 10L),
#                        makeNumericParam("subsample", lower = 0.5, upper = 1),
#                        makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
#                        makeNumericLearnerParam("gamma", lower = 0, upper = 5),
#                        makeNumericLearnerParam("eta", lower = 0, upper = 0.3)
# )
# 
# #set resampling strategy; use 5-fold Cross-Validation to measure improvements
# rdesc <- makeResampleDesc("CV", stratify = T, iters = 5L)
# #set stratify=T to ensure that distribution of target class
# #is maintained in the resampled data sets
# #iters: number of iterations
# 
# #set search optimization strategy
# #use random search (with 10 iterations) to find the optimal hyperparameter
# ctrl <- makeTuneControlRandom(maxit = 10L)
# 
# #set a parallel backend to ensure faster computation
# library(parallel)
# library(parallelMap)
# 
# parallelStartSocket(cpus = detectCores())
# 
# #parameter tuning
# mytune <- tuneParams(learner = lrn,
#                      task = traintask,
#                      resampling = rdesc,
#                      measures = acc,
#                      par.set = params,
#                      control = ctrl,
#                      show.info = T
# )
# 
# #mytune$x
# ##[Tune] Result: booster=gbtree; max_depth=5; min_child_weight=5.19;
# ##subsample=0.843; colsample_bytree=0.763; gamma=2.73; eta=0.133
# ##: acc.test.mean=0.9827714
# 
# #train XGBoost model using tuned hyperparameters
# lrn_tune <- setHyperPars(lrn, par.vals = mytune$x)
# xgmodel <- train(learner = lrn_tune, task = traintask)
# 
# #model prediction
# xgpred2_train <- predict(xgmodel, traintask)
# xgpred2_test <- predict(xgmodel, testtask)
# 
# #confusion matrix
# library(caret)
# confusionMatrix(xgpred2_test$data$response, xgpred2_test$data$truth)
# ##Accuracy : 0.9817
# ##F-Score : 0.8878
# 
# #stop parallelization
# parallelStop()
# 
# #need to retrain model using xgboost package with the same parameters
# #to get feature importance plot
# params <- list(booster = "gbtree",
#                objective = "binary:logistic",
#                eta = mytune$x[[7]],
#                gamma = mytune$x[[6]],
#                max_depth = mytune$x[[2]],
#                min_child_weight = mytune$x[[3]],
#                subsample = mytune$x[[4]],
#                colsample_bytree = mytune$x[[5]]
# )
# 
# xgb <- xgb.train(params = params,
#                  data = dtrain,
#                  nrounds = best_error_index,
#                  watchlist = list(train = dtrain, val = dtest),
#                  print_every_n = 50,
#                  early_stop_round = 10,
#                  maximize = F ,
#                  eval_metric = "error",
#                  eval_metric = 'auc'
# )
# 
# #view variable importance plot
# mat2 <- xgb.importance(feature_names = colnames(training_set), model = xgb)
# xgb.plot.importance(importance_matrix = mat2[0:20])
# xgb.ggplot.importance(importance_matrix = mat2[0:20])
# 
# #model prediction
# xgbpred2_train <- predict(xgb, dtrain)
# xgbpred2_test <- predict(xgb, dtest)
# 
# #train AUC value
# library(pROC)
# tr_label <- as.numeric(training_set$bad)
# par(pty = "s")
# roc_training <- roc(tr_label,
#                     xgbpred2_train,
#                     algorithm = 2,
#                     percent = T)
# plot(roc_training,
#      print.auc = T,
#      col = "blue",
#      print.auc.x = 40,
#      print.auc.y = 95)
# auc(roc_training)
# ##Area under the curve: 99.96%
# 
# #test AUC value
# ts_label <- as.numeric(test_set$bad)
# roc_test <- roc(response = ts_label,
#                 predictor = xgbpred2_test,
#                 algorithm = 2,
#                 percent = T)
# plot(roc_test,
#      print.auc = T,
#      col = "red",
#      print.auc.x = 80,
#      print.auc.y = 95,
#      add = T)
# auc(roc_test)
# ##Area under the curve: 98.31%