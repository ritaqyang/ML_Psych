library(caret) 
library(pROC)
library(MASS)
library(e1071)
library(class)
setwd("/Users/ritayang/A2/A2")

mydata = read.csv("mobilephone_1_training.csv") 
mydata$status24 = as.factor(mydata$status24)
mydata_test = read.csv("mobilephone_1_test.csv") 
mydata_test$status24 = as.factor(mydata_test$status24)


log.fit = glm(status24 ~ age + ssound + sglob + svmail + sinfor
              + sprice + spromo + sadver + sperson + simage + lintent,
              data = mydata,family = binomial )

lda.fit <- lda(status24 ~ age + ssound + sglob + svmail + sinfor + 
                 sprice + spromo + sadver + sperson + simage + lintent,
               data = mydata ) 

qda.fit <- qda(status24 ~ age + ssound + sglob + svmail + sinfor + 
                 sprice + spromo + sadver + sperson + simage + lintent,
               data = mydata ) 

nb.fit <- naiveBayes(status24 ~ age + ssound + sglob + svmail + sinfor + 
                       sprice + spromo + sadver + sperson + simage + lintent,
                     data = mydata ) 


log.prob.y.tt = predict(log.fit, mydata_test, type = "response")
log.pred.y.tt = ifelse(log.prob.y.tt>.5,1,0)
log.pred.y.tt = as.factor(log.pred.y.tt)
confusionMatrix(data = log.pred.y.tt, reference = mydata_test$status24, positive = "1")
log.roc.tt = roc(mydata_test$status24,log.prob.y.tt,auc=TRUE) 


lda.list.y.tt = predict(lda.fit,mydata_test)
lda.prob.y.tt = lda.list.y.tt$posterior
lda.prob.y.tt = lda.prob.y.tt[,2]
lda.pred.y.tt = lda.list.y.tt$class
confusionMatrix(data = lda.pred.y.tt, reference = mydata_test$status24, positive = "1")
lda.roc.tt = roc(mydata_test$status24,lda.prob.y.tt,auc=TRUE) 


qda.list.y.tt = predict(qda.fit,mydata_test)
qda.prob.y.tt = qda.list.y.tt$posterior
qda.prob.y.tt = qda.prob.y.tt[,2]
qda.pred.y.tt = qda.list.y.tt$class
confusionMatrix(data = qda.pred.y.tt, reference = mydata_test$status24, positive = "1")
qda.roc.tt = roc(mydata_test$status24,qda.prob.y.tt,auc=TRUE) 


nb.prob.y.tt = predict(nb.fit,mydata_test,type = "raw")
nb.prob.y.tt = nb.prob.y.tt[,2]
nb.pred.y.tt = predict(nb.fit,mydata_test,type = "class")
confusionMatrix(data = nb.pred.y.tt, reference = mydata_test$status24, positive = "1")
nb.roc.tt = roc(mydata_test$status24,nb.prob.y.tt,auc=TRUE)

# 6. KNN ----
## 6.1 Preprocess data ----
mydata.X = scale(mydata[,c(3:13)])
mydata.Y = mydata[,2]
mydata_test.X = scale(mydata_test[,c(3:13)])
mydata_test.Y = mydata_test[,2]


## 6.2 Run KNN
K=300
knn.pred.y.tt=knn(mydata.X, mydata_test.X, mydata.Y, K, prob=TRUE)
knn.prob.y.tt=attr(knn.pred.y.tt,"prob")

confusionMatrix(data = knn.pred.y.tt, reference = mydata_test.Y, positive = "1")
knn.roc.tt = roc(mydata_test.Y,knn.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
plot(knn.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
     ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
     auc.polygon=TRUE)


# 7. Model Comparison (AUC) ----
plot(log.roc.tt,legacy.axes=TRUE,col="black",lwd=4,
     ylab = "True Positive Rate",xlab = "False Positive Rate",
     main = "ROC",grid=TRUE)
plot(lda.roc.tt,col="red",   lwd=4,add=TRUE)
plot(qda.roc.tt,col="green", lwd=4,add=TRUE)
plot(nb.roc.tt ,col="purple",lwd=4,add=TRUE)
legend("bottomright",
       legend=c("LR","LDA","QDA","NB"),
       col=c("black","red","green","purple"),
       lwd=4)

cat("\n AUC_LR:  ", log.roc.tt$auc,
    "\n AUC_LDA: ", lda.roc.tt$auc,
    "\n AUC_QDA: ", qda.roc.tt$auc,
    "\n AUC_NB:  ", nb.roc.tt$auc)



# 7. Model Comparison (AUC) ----
plot(log.roc.tt,legacy.axes=TRUE,col="black",lwd=4,
     ylab = "True Positive Rate",xlab = "False Positive Rate",
     main = "ROC",grid=TRUE)
plot(lda.roc.tt,col="red",   lwd=4,add=TRUE)
plot(qda.roc.tt,col="green", lwd=4,add=TRUE)
plot(nb.roc.tt ,col="purple",lwd=4,add=TRUE)
plot(knn.roc.tt,col="blue",  lwd=4,add=TRUE)
legend("bottomright",
       legend=c("LR","LDA","QDA","NB","KNN"),
       col=c("black","red","green","purple","blue"),
       lwd=4)

cat("\n AUC_LR:  ", log.roc.tt$auc,
    "\n AUC_LDA: ", lda.roc.tt$auc,
    "\n AUC_QDA: ", qda.roc.tt$auc,
    "\n AUC_NB:  ", nb.roc.tt$auc,
    "\n AUC_KNN: ", knn.roc.tt$auc)

