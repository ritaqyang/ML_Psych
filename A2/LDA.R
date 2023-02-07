
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
# 3. LDA ----Linear Discriminant Analysis 
## 3.1 Run a model ----

lda.fit <- lda(status24 ~ age + ssound + sglob + svmail + sinfor + 
                 sprice + spromo + sadver + sperson + simage + lintent,
               data = mydata ) 

## 3.2 Additional Analysis ----
### on training sample ----
#predictive power 
lda.list.y.tr = predict(lda.fit) 
lda.prob.y.tr = lda.list.y.tr$posterior 
lda.prob.y.tr = lda.prob.y.tr[,2]
lda.pred.y.tr = lda.list.y.tr$class

confusionMatrix(data = lda.pred.y.tr, reference = mydata$status24, positive = "1") # obtain confusion matrix
lda.roc.tr = roc(mydata$status24,lda.prob.y.tr,auc=TRUE) # plot the ROC curve with AUC 
plot(lda.roc.tr,print.auc=TRUE,legacy.axes=TRUE,
     ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
     auc.polygon=TRUE,grid=TRUE)

### on test sample ----
lda.list.y.tt = predict(lda.fit,mydata_test)
lda.prob.y.tt = lda.list.y.tt$posterior
lda.prob.y.tt = lda.prob.y.tt[,2]
lda.pred.y.tt = lda.list.y.tt$class

confusionMatrix(data = lda.pred.y.tt, reference = mydata_test$buyer, positive = "1")
lda.roc.tt = roc(mydata_test$buyer,lda.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
plot(lda.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
     ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
     auc.polygon=TRUE,grid=TRUE)

