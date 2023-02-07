
#install.packages('caret') # for making confusion Matrix
#install.packages('pROC')  # for plotting ROC curve
#install.packages('MASS')  # for conducting LDA/QDA
#install.packages('e1071')  # for Naive Bayes
#install.packages('class')  # for conducting KNN  

# 1. Load libraries and data
library(caret) 
library(pROC)
library(MASS)
library(e1071)
library(class)
setwd("/Users/ritayang/A2/A2")

mydata = read.csv("mobilephone_1_training.csv") 
head(mydata)
mydata$status24 = as.factor(mydata$status24)
head(mydata)

mydata_test = read.csv("mobilephone_1_test.csv") 
head(mydata_test)
mydata_test$status24 = as.factor(mydata_test$status24)
head(mydata_test)
# 2. Logistic Regression ----
## 2.1 Run a model ----
log.fit = glm(status24 ~ age + ssound + sglob + svmail + sinfor
              + sprice + spromo + sadver + sperson + simage + lintent,
              data = mydata,family = binomial )
#binomial probability 
summary(log.fit)
# obtain estimates of exp(B) and their 95%CI 
# exp(b) = odds ratio 

Exp_b = exp(coef(log.fit))
Exp_b
CI = confint(log.fit)
CI
Exp_CI = exp(confint(log.fit))
Exp_CI
combined = cbind(Exp_b, Exp_CI)
combined

Exp_b_CI=exp(cbind(coef(log.fit), confint(log.fit))) 
Exp_b_CI

## 2.2 Additional Analysis ----
### on training sample ----
log.prob.y.tr = predict(log.fit, type = "response") # calculate individual probabilities

# type = "response" tells R to calculate P(Y = 1|X) for each observation
log.pred.y.tr = ifelse(log.prob.y.tr>.5,1,0) # calculate predicted values using .5 cutoff
log.pred.y.tr = as.factor(log.pred.y.tr)

confusionMatrix(data = log.pred.y.tr, reference = mydata$status24, positive = "1") # obtain confusion matrix
log.roc.tr = roc(mydata$status24,log.prob.y.tr,auc=TRUE) # plot the ROC curve with AUC 

plot(log.roc.tr,print.auc=TRUE, legacy.axes=TRUE,
     ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
     auc.polygon=TRUE,grid=TRUE)
# legacy.axes = TRUE if the x-axis must be plotted as increasing FPR(1-specificity)
# auc.polygon = TRUE if you want to color the area under the ROC curve. 



