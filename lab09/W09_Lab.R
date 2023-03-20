#===#===#===#===#===#===#===#===#===#===#
# PSYC 560                              #
# Week 09: Support Vector Machines      #
# Heungsun Hwang and Gyeongcheol Cho    #
#===#===#===#===#===#===#===#===#===#===#

#install.packages('e1071') # for Support vector machines 
#install.packages('caret') # for making confusion Matrix
#install.packages('pROC')  # for plotting ROC curve

# 0. Load libraries & data ---- 
  library(e1071) 
  library(caret)
  library(pROC)
 
  mydata = read.csv('Heart_training.csv')
  head(mydata)
  mydata$Sex = factor(mydata$Sex ,c(0,1), labels = c("female","male")) 
  mydata$ChestPain  = as.factor(mydata$ChestPain) 
  mydata$RestEcg = as.factor(mydata$RestEcg)
  mydata$Fbs = as.factor(mydata$Fbs)
  mydata$ExAng   = as.factor(mydata$ExAng) 
  mydata$Slope   = as.factor(mydata$Slope) 
  mydata$Thal    = as.factor(mydata$Thal) 
  mydata$HeartDisease    = factor(mydata$HeartDisease,c(0,1),labels = c("No","Yes")) 
  
  mydata.tt = read.csv('Heart_test.csv')
  head(mydata.tt)
  mydata.tt$Sex = factor(mydata.tt$Sex ,c(0,1), labels = c("female","male")) 
  mydata.tt$ChestPain  = as.factor(mydata.tt$ChestPain) 
  mydata.tt$RestEcg = as.factor(mydata.tt$RestEcg)
  mydata.tt$Fbs = as.factor(mydata.tt$Fbs)
  mydata.tt$ExAng   = as.factor(mydata.tt$ExAng) 
  mydata.tt$Slope   = as.factor(mydata.tt$Slope) 
  mydata.tt$Thal    = as.factor(mydata.tt$Thal) 
  mydata.tt$HeartDisease    = factor(mydata.tt$HeartDisease,c(0,1),labels = c("No","Yes")) 

# 2. Support Vector Classifier  ----
  ##2-1. Train the model ----
  Cs = 10^seq(from = 1,to = -1, length.out = 50)
  svc.cv = tune(svm, HeartDisease ~ ., data = mydata, kernel = "linear",
           ranges = list( cost = Cs ))
  plot(svc.cv) # Draw the plot of CVE against C hyper-parameter 
  summary(svc.cv)
  svc.fit = svm(HeartDisease ~ ., data = mydata, kernel = "linear", cost = svc.cv$best.parameters$cost, probability = TRUE)

  ##2-2. Test the model ----
  svc.pred.y.tt = predict(svc.fit, mydata.tt,decision.values=TRUE,probability = TRUE)
  head(svc.pred.y.tt)
  confusionMatrix(data = svc.pred.y.tt, reference = mydata.tt$HeartDisease, positive = "Yes")
  
  svc.prob.y.tt = attr(svc.pred.y.tt, "probabilities")
  head(svc.prob.y.tt)
  svc.prob.y.tt = svc.prob.y.tt[,1]
  head(svc.prob.y.tt)
  svc.roc.tt = roc(mydata.tt$HeartDisease,svc.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
  plot(svc.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)

# 3. Support Vector Machine with a polynomial kernel trick ----
  ##3-1. Train the model ----
  Cs = 10^seq(from = 1,to = -1, length.out = 50)
  svm.p.cv = tune(svm, HeartDisease ~ ., data = mydata, kernel = "polynomial",
                ranges = list(degree = c(1,2,3), cost = Cs))
  plot(svm.p.cv, nlevels = 10) 
  summary(svm.p.cv)
  svm.p.fit = svm(HeartDisease ~ ., data = mydata, kernel = "polynomial", 
                 cost = svm.p.cv$best.parameters$cost, degree = svm.p.cv$best.parameters$degree,
                 probability = TRUE)
  ##3-2. Test the model ----
  svm.p.pred.y.tt = predict(svm.p.fit, mydata.tt,decision.values=TRUE,probability = TRUE)
  head(svm.p.pred.y.tt)
  confusionMatrix(data = svm.p.pred.y.tt, reference = mydata.tt$HeartDisease, positive = "Yes")
  
  svm.p.prob.y.tt = attr(svm.p.pred.y.tt, "probabilities")
  head(svm.p.prob.y.tt)
  svm.p.prob.y.tt = svm.p.prob.y.tt[,1]
  head(svm.p.prob.y.tt)
  svm.p.roc.tt = roc(mydata.tt$HeartDisease,svm.p.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
  plot(svm.p.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE) 
  
# 4. Support Vector Machine with a radial kernel trick ----
  ##4-1. Train the model ----
  Cs = 10^seq(from = 1,to = -1, length.out = 50)
  svm.r.cv = tune(svm, HeartDisease ~ ., data = mydata, kernel = "radial",
                  ranges = list(gamma = c(10^-1,10^-2,10^-3), cost = Cs))
  plot(svm.r.cv, nlevels = 10)
  svm.r.fit = svm(HeartDisease ~ ., data = mydata, kernel = "radial", 
                  cost = svm.r.cv$best.parameters$cost, gamma = svm.r.cv$best.parameters$gamma,
                  probability = TRUE)
  
  ##4-2. Test the tree ----
  svm.r.pred.y.tt = predict(svm.r.fit, mydata.tt,decision.values=TRUE,probability = TRUE)
  head(svm.r.pred.y.tt)
  confusionMatrix(data = svm.r.pred.y.tt, reference = mydata.tt$HeartDisease, positive = "Yes")
  
  svm.r.prob.y.tt = attr(svm.r.pred.y.tt, "probabilities")
  head(svm.r.prob.y.tt)
  svm.r.prob.y.tt = svm.r.prob.y.tt[,1]
  head(svm.r.prob.y.tt)
  svm.r.roc.tt = roc(mydata.tt$HeartDisease,svm.r.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
  plot(svm.r.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE) 
  
  # 6. Model Comparison (ROC & AUC) ----
  plot(svc.roc.tt,legacy.axes=TRUE,col="black",lwd=4,
       ylab = "True Positive Rate",xlab = "False Positive Rate",
       main = "ROC",grid=TRUE)
  plot(svm.p.roc.tt,legacy.axes=TRUE,col="red",lwd=4,add=TRUE)
  plot(svm.r.roc.tt,legacy.axes=TRUE,col="green",lwd=4,add=TRUE)
  legend("bottomright",legend=c("SVC","SVM(P)","SVM(R)"),col=c("black","red","green"),lwd=4)
  
  cat("\n AUC(SVC)            : ",svc.roc.tt$auc,
      "\n AUC(SVM(Polynomial)): ",svm.p.roc.tt$auc,
      "\n AUC(SVM(Radial))    : ",svm.r.roc.tt$auc)