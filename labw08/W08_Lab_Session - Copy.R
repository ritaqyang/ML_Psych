#===#===#===#===#===#===#===#===#===#===#
# PSYC 560                              #
# Week 08: Tree-based methods           #
# Heungsun Hwang and Gyeongcheol Cho    #
#===#===#===#===#===#===#===#===#===#===#

#install.packages('tree') # for tree methods
#install.packages('caret') # for making confusion Matrix
#install.packages('pROC')  # for plotting ROC curve
#install.packages('randomForest')  # for Bagging and Random Forest algorithms 
#install.packages('gbm')  # for boosting

# 0. Load libraries & data ---- 
  library(tree) 
  library(caret)
  library(pROC)
  library(randomForest)
  library(gbm)
  setwd('C:/Users/cheol/Dropbox/Mcgill/Lecture/Machine_Learning_Hwang/W06_Tree_Methods')

  # For regression
  mydata = read.csv('Hitters2_training.csv')
  View(mydata)
  mydata$League =as.factor(mydata$League) 
  mydata$Division = as.factor(mydata$Division)
  mydata$NewLeague =as.factor(mydata$NewLeague)
  
  mydata_test = read.csv('Hitters2_test.csv')
  mydata_test$League =as.factor(mydata_test$League) 
  mydata_test$Division = as.factor(mydata_test$Division)
  mydata_test$NewLeague =as.factor(mydata_test$NewLeague)
  
  # For classification
  mydata2 = read.csv('Heart_training.csv')
  head(mydata2)
  mydata2$Sex = factor(mydata2$Sex ,c(0,1), labels = c("female","male")) 
  mydata2$ChestPain  = as.factor(mydata2$ChestPain) 
  mydata2$RestEcg = as.factor(mydata2$RestEcg)
  mydata2$Fbs = as.factor(mydata2$Fbs)
  mydata2$ExAng   = as.factor(mydata2$ExAng) 
  mydata2$Slope   = as.factor(mydata2$Slope) 
  mydata2$Thal    = as.factor(mydata2$Thal) 
  mydata2$HeartDisease    = factor(mydata2$HeartDisease,c(0,1),labels = c("No","Yes")) 
  
  mydata2.tt = read.csv('Heart_test.csv')
  head(mydata2.tt)
  mydata2.tt$Sex = factor(mydata2.tt$Sex ,c(0,1), labels = c("female","male")) 
  mydata2.tt$ChestPain  = as.factor(mydata2.tt$ChestPain) 
  mydata2.tt$RestEcg = as.factor(mydata2.tt$RestEcg)
  mydata2.tt$Fbs = as.factor(mydata2.tt$Fbs)
  mydata2.tt$ExAng   = as.factor(mydata2.tt$ExAng) 
  mydata2.tt$Slope   = as.factor(mydata2.tt$Slope) 
  mydata2.tt$Thal    = as.factor(mydata2.tt$Thal) 
  mydata2.tt$HeartDisease    = factor(mydata2.tt$HeartDisease,c(0,1),labels = c("No","Yes")) 
  
# 1. Regression tree ---- 
  ##1-1. Train the tree ----
  rtree.fit = tree(Salary ~. ,mydata)  
  rtree.fit
  plot(rtree.fit)
  text(rtree.fit)
  summary(rtree.fit)
  
  ##1-2. Prune the tree ----
  rtree.cv = cv.tree(rtree.fit, K = 5)
  plot(rtree.cv$size,rtree.cv$dev, type ="o", xlab = "# of terminal nodes", ylab = "CVE")  
    # type = "p"(default) for points plot
    #      = "l" for line plots
    #      = "b" or "o" for line & points
  rtree.cv
    list_CVE=cbind(rtree.cv$size,rtree.cv$dev)
    colnames(list_CVE)=c('# of terminal nodes','CVE')
    list_CVE
    
  rtree.fit.pruned = prune.tree(rtree.fit, best = 8)
  rtree.fit.pruned
  
  plot(rtree.fit.pruned)
  text(rtree.fit.pruned)
  
  #plot(rtree.fit)
  #text(rtree.fit)
  
  ##1-3. Test the tree ----
  rtree.pred.y.tt        = predict(rtree.fit, mydata_test)
  rtree.pred.y.tt.pruned = predict(rtree.fit.pruned, mydata_test)
  rtree.original.MSE = mean((rtree.pred.y.tt-mydata_test$Salary)^2)
  rtree.pruned.MSE = mean((rtree.pred.y.tt.pruned-mydata_test$Salary)^2)
  cat("\n MSE_original(test): ", rtree.original.MSE,
      "\n MSE_pruned(test)  : ", rtree.pruned.MSE)

# 2. Classification tree ----
  ##2-1. Train the tree ----
  ctree.fit = tree(HeartDisease ~ . ,mydata2)  
  summary(ctree.fit)
  plot(ctree.fit)
  text(ctree.fit)
  ctree.fit
  
  ##2-2. Prune the tree ----
  ctree.cv = cv.tree(ctree.fit, K = 5)
  plot(ctree.cv$size,ctree.cv$dev, type ="o", xlab = "# of terminal nodes", ylab = "CVE")    
  ctree.cv
    
  ctree.fit.pruned = prune.tree(ctree.fit, best = 4)
  plot(ctree.fit.pruned)
  text(ctree.fit.pruned)
  ctree.fit.pruned

  ##2-3. Test the tree ----
  ctree.pred.y.tt = predict(ctree.fit, mydata2.tt, type = "class")
  confusionMatrix(data = ctree.pred.y.tt, reference = mydata2.tt$HeartDisease, positive = "Yes")
  ctree.prob.y.tt = predict(ctree.fit, mydata2.tt, type = "vector")
  ctree.roc.tt = roc(mydata2.tt$HeartDisease,ctree.prob.y.tt[,2],auc=TRUE) # plot the ROC curve with AUC 
  plot(ctree.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)

  ctree.pred.y.tt.pruned = predict(ctree.fit.pruned, mydata2.tt, type = "class")
  confusionMatrix(data = ctree.pred.y.tt.pruned, reference = mydata2.tt$HeartDisease, positive = "Yes")
  ctree.prob.y.tt.pruned = predict(ctree.fit.pruned, mydata2.tt, type = "vector")
  ctree.roc.tt.pruned = roc(mydata2.tt$HeartDisease,ctree.prob.y.tt.pruned[,2],auc=TRUE) # plot the ROC curve with AUC 
  plot(ctree.roc.tt.pruned,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)
  
# 3. Bagging ----
  ##3-1. Train the bagged trees
  bag.fit = randomForest(HeartDisease ~., mydata2, mtry = 13,  importance = TRUE)
    # ntree=500
  bag.fit  
  varImpPlot(bag.fit, type = 2)   
  
  ##3-2. Test the bagged trees
  bag.pred.y.tt = predict(bag.fit,mydata2.tt)
  confusionMatrix(data = bag.pred.y.tt, reference = mydata2.tt$HeartDisease, positive = "Yes")
  bag.prob.y.tt = predict(bag.fit,mydata2.tt,type="prob")
  bag.roc.tt = roc(mydata2.tt$HeartDisease,bag.prob.y.tt[,2],auc=TRUE) # plot the ROC curve with AUC 
  plot(bag.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)

# 4. Random Forest ----  
  ##4-1. Train the random forest ----
  RF.fit = randomForest(HeartDisease ~., mydata2, importance = TRUE)
    #mtry=floor(sqrt(number of predictors))
  RF.fit  
  varImpPlot(RF.fit, type = 2)

  ##4-2. Test the random forest ----  
  RF.pred.y.tt = predict(RF.fit,mydata2.tt)
  confusionMatrix(data = RF.pred.y.tt, reference = mydata2.tt$HeartDisease, positive = "Yes")
  RF.prob.y.tt = predict(RF.fit,mydata2.tt,type="prob")
  RF.roc.tt = roc(mydata2.tt$HeartDisease,RF.prob.y.tt[,2],auc=TRUE) # plot the ROC curve with AUC 
  plot(RF.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)
 
# 5. Boosting ----
  mydata3=mydata2
  mydata3$HeartDisease=as.numeric(mydata3$HeartDisease)
  head(mydata3)
  mydata3$HeartDisease=mydata3$HeartDisease-1
  
  ##5-1. Train the boosted trees ----
  boost.fit = gbm(HeartDisease ~., mydata3, 
                  distribution = "bernoulli", 
                  cv.folds = 5)
    # distribution = "gaussian" 
  gbm.perf(boost.fit, method = "cv")
  boost.fit
    # min(boost.fit$cv.error)
  summary(boost.fit)
  
  ##5-2. Test the boosted trees ----
  boost.prob.y.tt = predict(boost.fit,mydata2.tt,type = "response")
  boost.pred.y.tt = ifelse(boost.prob.y.tt>.5,1,0)
  boost.pred.y.tt = factor(boost.pred.y.tt,c(0,1),c("No","Yes"))
  confusionMatrix(data = boost.pred.y.tt, reference = mydata2.tt$HeartDisease, positive = "Yes")
  boost.roc.tt = roc(mydata2.tt$HeartDisease,boost.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
  plot(boost.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)
  
# 6. Model Comparison (ROC & AUC) ----
  plot(ctree.roc.tt,legacy.axes=TRUE,col="black",lwd=4,
       ylab = "True Positive Rate",xlab = "False Positive Rate",
       main = "ROC",grid=TRUE)
  plot(bag.roc.tt,legacy.axes=TRUE,col="red",lwd=4,add=TRUE)
  plot(RF.roc.tt,legacy.axes=TRUE,col="green",lwd=4,add=TRUE)
  plot(boost.roc.tt,legacy.axes=TRUE,col="purple",lwd=4,add=TRUE)
  legend("bottomright",legend=c("tree","bagging","random forest","boosting"),col=c("black","red","green","purple"),lwd=4)
  
  cat("\n AUC(tree)         : ",ctree.roc.tt$auc,
      "\n AUC(bagging)      : ",bag.roc.tt$auc,
      "\n AUC(random forest): ",RF.roc.tt$auc,
      "\n AUC(boosting)     : ",boost.roc.tt$auc)