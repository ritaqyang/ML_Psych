#===#===#===#===#===#===#===#===#===#===#
# PSYC 560                              #
# Week #4: Classification I             #
# Instructor: Heungsun Hwang            #
# TA: Gyeongcheol Cho                   #
#===#===#===#===#===#===#===#===#===#===#

#install.packages('caret') # for making confusion Matrix
#install.packages('pROC')  # for plotting ROC curve

# 1. Load packages and import data
  library(caret) 
  library(pROC)
  setwd('C:\\Users\\cheol\\Dropbox\\Mcgill\\Lecture\\Machine_Learning_Hwang\\2023_Winter\\W04_Logistic_Regression')
        
  mydata = read.csv("BBB_training.csv") 
  mydata$buyer = as.factor(mydata$buyer)
#  mydata$gender = as.factor(mydata$gender)
  head(mydata)
    
  mydata_test = read.csv("BBB_test.csv") 
  mydata_test$buyer = as.factor(mydata_test$buyer)
#  mydata_test$gender = as.factor(mydata_test$gender)
  head(mydata_test)

# 2. Logistic Regression ----
  ## 2.1 Run a model ----
  log.fit = glm(buyer ~ gender + last + book + art + child
                         + youth + cook + do_it + reference + geog,
                         data = mydata , family = binomial)
  summary(log.fit)
  Exp_b_CI=exp(cbind(coef(log.fit), confint(log.fit))) # obtain estimates of exp(B) and their 95%CI 
  Exp_b_CI
  
  ## 2.2 Additional Analysis ----
  ### on training sample ----
  log.prob.y.tr = predict(log.fit, type = "response") # calculate individual probabilities
    # type = "response" tells R to calculate P(Y = 1|X) for each observation
  log.pred.y.tr = ifelse(log.prob.y.tr>.5,1,0) # calculate predicted values using .5 cutoff
  log.pred.y.tr = as.factor(log.pred.y.tr)
  confusionMatrix(data = log.pred.y.tr, reference = mydata$buyer, positive = "1") # obtain confusion matrix
  log.roc.tr = roc(mydata$buyer,log.prob.y.tr,auc=TRUE) # plot the ROC curve with AUC 
  plot(log.roc.tr,print.auc=TRUE, legacy.axes=TRUE,
        ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
        auc.polygon=TRUE,grid=TRUE)
    # legacy.axes = TRUE if the x-axis must be plotted as increasing FPR(1-specificity)
    # auc.polygon = TRUE if you want to color the area under the ROC curve. 
  
  ### on test sample ----
  log.prob.y.tt = predict(log.fit, mydata_test, type = "response")
  log.pred.y.tt = ifelse(log.prob.y.tt>.5,1,0)
  log.pred.y.tt = as.factor(log.pred.y.tt)
  confusionMatrix(data = log.pred.y.tt, reference = mydata_test$buyer, positive = "1")
  log.roc.tt = roc(mydata_test$buyer,log.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
  plot(log.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)