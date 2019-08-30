#https://www.rdocumentation.org/packages/scorecard/versions/0.1.4
#https://www.r-pkg.org/pkg/scorecard

#install all package
install.packages("here")
install.packages("readr")
install.packages("plyr")
install.packages("lubridate")
install.packages("dplyr")
install.packages("chron")
install.packages("caTools")
install.packages("scorecard")

library(here)
library(readr)
library(plyr)
library(lubridate)
library(dplyr)
library(chron)
library(caTools)
library(scorecard)

#load CSV file
dataset_path <- here::here("LoanStats_2018Q1_ds.csv")
data <- read.csv(dataset_path)

#create variable
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

#drop '%'
data$int_rate <- as.numeric(sub("%", "", data$int_rate))
data$revol_util <- as.numeric(sub("%", "", data$revol_util))

#drop variable
data2 = subset(data, select = -c(issue_d, earliest_cr_line, id, member_id, emp_title, sec_app_earliest_cr_line, loan_status, last_pymnt_d, next_pymnt_d, last_credit_pull_d))

#filter variable via missing rate, iv, identical value rate
dt_f = var_filter(data2, y = "bad")

#breaking dt into train and test
dt_list = split_df(dt_f, y = "bad", ratio = 0.7, seed = 30)
train = dt_list$train; test = dt_list$test;
label_list = lapply(dt_list, function(x) x$bad)

#woe binning on train
bins = woebin(train, y = "bad")
plotlist = woebin_plot(bins)

#save binning plot
for (i in 1:length(plotlist)) {
  ggplot2::ggsave(paste0(names(plotlist[i]), ".png"), 
                  plotlist[[i]], 
                  path = here::here("binning_plots/"),
                  width = 15, 
                  height = 9, 
                  units="cm" )
}

#binning adjustment
#specify breaks manually
breaks_adj = list(total_rec_int = c(200, 400, 5200),
                  il_util = c(30, 60, 75, 85, "Inf%,%missing"),
                  age_of_oldest_cr_line_in_mth = c(90, 210, 400),
                  mo_sin_old_il_acct = c("missing%,%30", 70, 190),
                  mths_since_recent_revol_delinq = c(40, 60),
                  total_bal_ex_mort = c(5000, 45000)) 
bins_adj = woebin(train, y = "bad", breaks_list = breaks_adj)
plotlist = woebin_plot(bins_adj)

#var not use
#all_util, annual_inc_joint, bc_util, dti, emp_length, funded_amnt, 
#funded_amnt_inv, last_pymnt_amnt, loan_amnt, mths_since_last_major_derog,
#mths_since_last_record, mths_since_recent_bc_dlq, num_il_tl,
#num_rev_accts, num_sats, open_acc, out_prncp,out_prncp_inv, pct_tl_nvr_dlq,
#percent_bc_gt_75, revol_bal_joint, revol_util, sec_app_collections_12_mths_ex_med,
#sec_app_inq_last_6mths, sec_app_mort_acc, sec_app_mths_since_last_major_derog,
#sec_app_num_rev_accts, sec_app_open_acc, sec_app_revol_util,
#tot_coll_amt, total_acc, total_bal_il, total_rec_int, mths_since_rcnt_il, mths_since_recent_bc

#converting train and test into woe values
train_woe = woebin_ply(train, bins_adj)
test_woe = woebin_ply(test, bins_adj)

#drop var that have nonlinear pattern in binning and score (after checking summary of 'card')
train2 <- select(train,
                 -c(all_util, annual_inc_joint, bc_util, dti, emp_length, funded_amnt,
                    funded_amnt_inv, last_pymnt_amnt, loan_amnt, mths_since_last_major_derog,
                    mths_since_last_record, mths_since_recent_bc_dlq, num_il_tl,
                    num_rev_accts, num_sats, open_acc, out_prncp,out_prncp_inv, pct_tl_nvr_dlq,
                    percent_bc_gt_75, revol_bal_joint, revol_util, sec_app_collections_12_mths_ex_med,
                    sec_app_inq_last_6mths, sec_app_mort_acc, sec_app_mths_since_last_major_derog,
                    sec_app_num_rev_accts, sec_app_open_acc, sec_app_revol_util, tot_coll_amt, 
                    total_acc, total_bal_il, total_rec_int, mths_since_rcnt_il, mths_since_recent_bc, #var that have weird pattern in binning
                    int_rate, sub_grade, total_pymnt_inv, total_il_high_credit_limit, #var that are higherly correlated
                    verification_status, revol_bal, dti_joint, tot_hi_cred_lim, #var that have score pattern cannot be explain
                    open_rv_12m, num_tl_op_past_12m, total_bal_ex_mort #var that are not significant
                    ))
train_woe = woebin_ply(train2, bins_adj)
test_woe = woebin_ply(test, bins_adj)

#glm
m1 = glm(bad ~ ., family = "binomial", data = train_woe)
vif(m1, merge_coef = TRUE)
summary(m1)

#select a formula-based model by AIC
m_step = step(m1, direction = "both", trace = FALSE)
m2 = eval(m_step$call)
vif(m2, merge_coef = TRUE)
summary(m2)

# # Adjusting for oversampling (support.sas.com/kb/22/601.html)
# library(data.table)
# p1=0.03 # bad probability in population 
# r1=0.3 # bad probability in sample dataset
# dt_woe = copy(dt_woe_list$train)[, weight := ifelse(creditability==1, p1/r1, (1-p1)/(1-r1) )][]
# fmla = as.formula(paste("creditability ~", paste(names(coef(m2))[-1], collapse="+")))
# m3 = glm(fmla, family = binomial(), data = dt_woe, weights = weight)

# score
card = scorecard(bins_adj, m2)
sink("scorecard summary.txt")
print(card)
sink()

#var consider to drop
#round 1
#verification_status: verified has lower score than not verified; does not make sense
#revol_bal: higher the revol_bal lower the risk; expect the reverse
#dti_joint: cannot explain
#tot_hi_cred_lim: higher the limit lower the risk; tricky to explain

#round 2 (not significant var)
#open_rv_12m
#num_tl_op_past_12m
#total_bal_ex_mort

#performance
# predicted proability
train_pred = predict(m2, train_woe, type = 'response')
test_pred = predict(m2, test_woe, type = 'response')

# ks & roc plot
train_perf = perf_eva(label = train$bad, 
                      pred = train_pred,
                      title = "train")
test_perf = perf_eva(label = test$bad, 
                     pred = test_pred, 
                     title = "test")

# credit score, only_total_score = TRUE
train_score = scorecard_ply(train, card, print_step = 0)
test_score = scorecard_ply(test, card, print_step = 0)

# psi
perf_psi(
  score = list(train = train_score, test = test_score),
  label = list(train = train$bad, test = test$bad),
  x_limits = c(250, 700),
  x_tick_break = 50
)