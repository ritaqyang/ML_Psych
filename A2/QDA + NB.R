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

# 4. QDA ----
## 4.1 Run a model ----
qda.fit <- qda(status24 ~ age + ssound + sglob + svmail + sinfor + 
                     sprice + spromo + sadver + sperson + simage + lintent,
                   data = mydata ) 
## 4.2 Additional Analysis ----
### on training sample ----
qda.list.y.tr = predict(qda.fit)
qda.prob.y.tr = qda.list.y.tr$posterior
qda.prob.y.tr = qda.prob.y.tr[,2]
qda.pred.y.tr = qda.list.y.tr$class

confusionMatrix(data = qda.pred.y.tr, reference = mydata$status24, positive = "1") # obtain confusion matrix
qda.roc.tr = roc(mydata$status24,qda.prob.y.tr,auc=TRUE) # plot the ROC curve with AUC 
plot(qda.roc.tr,print.auc=TRUE,legacy.axes=TRUE,
     ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
     auc.polygon=TRUE,grid=TRUE)

# 5. Naive Bayes ----
## 5.1 Run a model ----
nb.fit <- naiveBayes(status24 ~ age + ssound + sglob + svmail + sinfor + 
                       sprice + spromo + sadver + sperson + simage + lintent,
                     data = mydata ) 

## 5.2 Additional Analysis ----
### on training sample ----
nb.prob.y.tr = predict(nb.fit,mydata,type = "raw")
nb.prob.y.tr = nb.prob.y.tr[,2]
nb.pred.y.tr = predict(nb.fit,mydata,type = "class")

confusionMatrix(data = nb.pred.y.tr, reference = mydata$status24, positive = "1") # obtain confusion matrix
nb.roc.tr = roc(mydata$status24,nb.prob.y.tr,auc=TRUE) # plot the ROC curve with AUC 
plot(nb.roc.tr,print.auc=TRUE,legacy.axes=TRUE,
     ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
     auc.polygon=TRUE,grid=TRUE)
