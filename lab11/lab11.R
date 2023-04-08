#===#===#===#===#===#===#===#===#===#===#
#  PSYC 560                             #
#  Week 11: Deep learning               #
#  Gyeongcheol Cho and Heungsun Hwang   #
#===#===#===#===#===#===#===#===#===#===#

## Packages for deep learning 
#install.packages('torch')
#install.packages('luz')    
#install.packages('glmnet') # for regularized regression
#install.packages('randomForest')  # for Bagging and Random Forest algorithms 
#install.packages('gbm')  # for boosting
library(torch)
library(luz) 
library(glmnet)
library(randomForest)
library(gbm)
# ########################################################################## #
# 1. Regression Problem ----
## 1.1. Data preprocessing  ----

mydata = read.csv('Hitters_training.csv')
head(mydata)
mydata$League =as.factor(mydata$League) 
mydata$Division = as.factor(mydata$Division)
mydata$NewLeague =as.factor(mydata$NewLeague)

head(mydata)
mydata.tt = read.csv('Hitters_test.csv')
mydata.tt$League =as.factor(mydata.tt$League) 
mydata.tt$Division = as.factor(mydata.tt$Division)
mydata.tt$NewLeague =as.factor(mydata.tt$NewLeague)

N=nrow(mydata)
N.va=round(N*.3)
N.tr=N-N.va
N.tt=nrow(mydata.tt)

id.vali = sample(1:N,N.va) 
mydata.va = mydata[id.vali,]
mydata.tr = mydata[-id.vali,]
#mydata.tt = mydata.tt

mydata.tr.X=model.matrix(Salary~.-1,data=mydata.tr)
mydata.tr.Y=mydata.tr$Salary
mydata.va.X=model.matrix(Salary~.-1,data=mydata.va)
mydata.va.Y=mydata.va$Salary
mydata.tt.X=model.matrix(Salary~.-1,data=mydata.tt)
mydata.tt.Y=mydata.tt$Salary

mean.tr.X=colMeans(mydata.tr.X)
sd.tr.X=apply(mydata.tr.X,2,sd)
mydata.tr.X=(mydata.tr.X-matrix(1,N.tr,1)%*%t(mean.tr.X))/(matrix(1,N.tr,1)%*%sd.tr.X)
mydata.va.X=(mydata.va.X-matrix(1,N.va,1)%*%t(mean.tr.X))/(matrix(1,N.va,1)%*%sd.tr.X)
mydata.tt.X=(mydata.tt.X-matrix(1,N.tt,1)%*%t(mean.tr.X))/(matrix(1,N.tt,1)%*%sd.tr.X)

mydata.tr.X.torch=torch_tensor(mydata.tr.X,torch_float())
mydata.tr.Y.torch=torch_tensor(mydata.tr.Y,torch_float())
mydata.va.X.torch=torch_tensor(mydata.va.X,torch_float())
mydata.va.Y.torch=torch_tensor(mydata.va.Y,torch_float())

data.tr = list(mydata.tr.X.torch, mydata.tr.Y.torch)
data.va = list(mydata.va.X.torch, mydata.va.Y.torch)

## 1.2. Deep feedforward neural networks ----
### 1.2.1. Specifying a neural network ----
DL.R = nn_module(
  initialize = function() {
    self$hidden   = nn_linear(20, 64) #20 = number of variables 
    self$act.relu = nn_relu()
    self$dropout  = nn_dropout(0.1) #dropout rate given 
    self$output   = nn_linear(64, 1) #64- number of units per hidden layer, 1 hidden layer 
    #self$act.sigmoid = nn_sigmoid() for a binary  classification problem
    #self$act.softmax = nn_log_softmax() for a classification problem with multiple classes
  },
  forward = function(input) {
    input %>%
      self$hidden() %>%
      self$act.relu() %>%
      self$dropout()  %>%
      self$output()
  }
)%>%
  setup(
    loss = nn_mse_loss(), 
    # nn_bce_loss() for a binary  classification problem
    # nn_nll_loss() for a classification problems with multiple classes
    optimizer = optim_rmsprop, # advanced SGD
    #metrics = list(luz_metric_mae())
    # luz_metric_mae(): MAE
    # luz_metric_rmse(): RMSE
  )%>%
  set_opt_hparams(weight_decay = 0)

DL.R()


### 1.2.2. Training ----
DL.R.fit = 
  fit(DL.R,
      data = data.tr,
      valid_data = data.va,
      epochs = 100,
      dataloader_options = list(batch_size = 32),
      #callback =  luz_callback_early_stopping(monitor = "valid_loss",mode = "min",patience = 3),
      #verbose = FALSE
  )
DL.R.fit
plot(DL.R.fit)


### 1.2.3. Testing ----
DL.R.pred.y.tt = predict(DL.R.fit,mydata.tt.X) %>% as.numeric
DL.MSE = mean((DL.R.pred.y.tt-mydata.tt.Y)^2)
DL.MSE

## 1.3. Other ML methods ----
### 1.3.1.  Linear Regression ----
lm.fit = lm(Salary ~ ., mydata)
#summary(lm.fit)
lm.pred.y.tt = predict(lm.fit,mydata.tt)
lm.MSE = mean((lm.pred.y.tt - mydata.tt.Y)^2)

## 1.3.2. Ridge regression ----
mydata.X=data.matrix(mydata[,colnames(mydata)!="Salary"])
mydata.Y=mydata$Salary
mydata.tt.X=data.matrix(mydata.tt[,colnames(mydata.tt)!="Salary"])
lambdas = 10^seq(from = 3,to = -2, length.out = 100)
ridge.cv = cv.glmnet(mydata.X, mydata.Y,
                     alpha = 0, lambda = lambdas,
                     nfold = 10)
#plot(lambdas[100:1],ridge.cv$cvm[100:1],xlab="lambda",ylab="MSE")
ridge.lam=ridge.cv$lambda.min
ridge.fit= glmnet(mydata.X, mydata.Y,
                  alpha = 0, lambda = ridge.lam)
ridge.pred.y.tt = predict(ridge.fit,newx = mydata.tt.X)
ridge.MSE = mean((ridge.pred.y.tt - mydata.tt.Y)^2)

## 1.3.3. Random Forest ----
RF.fit = randomForest(Salary ~., mydata, importance = TRUE)
#RF.fit  
#varImpPlot(RF.fit, type = 1)
RF.pred.y.tt = predict(RF.fit,mydata.tt)
RF.MSE = mean((RF.pred.y.tt - mydata.tt.Y)^2)

## 1.3.4. Boosting ----
boost.fit = gbm(Salary ~., mydata, 
                distribution = "gaussian", 
                cv.folds = 10)
#summary(boost.fit)
#gbm.perf(boost.fit, method = "cv")
boost.pred.y.tt = predict(boost.fit,mydata.tt)
boost.MSE = mean((boost.pred.y.tt - mydata.tt.Y)^2)

## 1.4. Comparison ----
cat("\n MSE_DL:    ", DL.MSE,
    "\n MSE_LR:    ", lm.MSE,
    "\n MSE_Ridge: ", ridge.MSE,
    "\n MSE_RF:    ", RF.MSE,
    "\n MSE_Boost: ", boost.MSE)

# 2. Classification Problem ----
library(torch)
library(luz) 
## 2.1. Data pre-processing ----

N.bit = 8  
N.rows= 28
N.columns= 28

Data = read.csv('mnist_train.csv')
Data.tt = read.csv('mnist_test.csv')
View(Data)
head(data)
locY=1

focal_id=1
Max.depth=2^N.bit-1        
image.mat.ind=matrix(as.numeric(Data[focal_id,-1]), nrow = N.rows, ncol = N.columns)
image.mat.ind=image.mat.ind[,N.columns:1]
image(1:N.rows, 1:N.columns, image.mat.ind, col=gray((Max.depth:0)/Max.depth))
#t(image.mat.ind[1:N.columns,N.columns:1])

N.ratio = .2   
N=nrow(Data)
N.vali=round(N*N.ratio)
id.vali = sample(1:N,N.vali)
Data.tr=Data[-id.vali,]
Data.va=Data[id.vali,]

Data.DL.C = dataset(
  name = "Data for DL (Classification)",
  initialize = function(mydata,locY,N.bit) {
    # Numeric predictors
    Max.depth=2^N.bit-1 
    mydata.X=mydata[,-locY]/Max.depth
    mydata.X=model.matrix(~.-1,mydata.X)
    mydata.Y=mydata[,locY]
    self$x = mydata.X %>% as.matrix() %>% torch_tensor()
    self$y = mydata.Y %>% as.factor %>% as.integer# %>% torch_tensor
  },
  .getitem = function(idx) {
    x = self$x[idx,]
    y = self$y[idx]
    list(x = x,y = y)
  },
  .length = function() {
    self$y%>% length
  })
Data.tr.DL = Data.DL.C(Data.tr,locY,N.bit)
Data.va.DL = Data.DL.C(Data.va,locY,N.bit)
Data.tt.DL = Data.DL.C(Data.tt,locY,N.bit)

## 2.2. Specifying a neural network ----
DL.C = nn_module(
  initialize = function() {
    self$linear1 = nn_linear(28*28, 256) #pixels are 28 by 28, 256 hidden units in first layer 
    self$act.relu = nn_relu()
    self$drop4   = nn_dropout(p = 0.4)
    
    self$linear2 = nn_linear(256,   128)
    self$drop3   = nn_dropout(p = 0.3)
    
    self$linear3 = nn_linear(128,   10)
    self$act.softmax = nn_log_softmax(1) #nn_softmax() is susceptible to numeric errors
  },
  forward = function(input) {
    input %>%
      self$linear1() %>%
      self$act.relu() %>%
      self$drop4() %>%
      
      self$linear2() %>%
      self$act.relu() %>%
      self$drop3() %>%
      
      self$linear3() %>%  
      self$act.softmax()
  }
)%>%
  setup(
    loss =nn_nll_loss(),
    # nn_mse_loss(), for a regression problem  
    # nn_bce_loss(), for a binary  classification problem
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_accuracy())
  )
DL.C()

## 2.3. Training ----
DL.C.fit = 
  fit(DL.C,
      data = Data.tr.DL,
      valid_data = Data.va.DL,
      epochs = 10,
      dataloader_options = list(batch_size = 256),
      #callback =  luz_callback_early_stopping(monitor = "valid_acc",mode = "max",  patience = 2),
      #verbose = FALSE
  )
plot(DL.C.fit)

#N.epochs=length(DL.C.fit[["records"]][["metrics"]][["valid"]])
#List.VE = sapply(1:N.epochs, function(x) DL.C.fit[["records"]][["metrics"]][["valid"]][[x]]$acc)
#max.VE = max(List.VE)

### 2.4.  Testing ----
DL.C.logprob.y.tt.tensor=predict(DL.C.fit,Data.tt.DL)
DL.C.pred.y.tt =torch_argmax(DL.C.logprob.y.tt.tensor,dim = 2) %>% as_array()
Data.tt.Y = sapply(1:length(Data.tt.DL), function(x) Data.tt.DL[x][[2]])
Data.tt.Y
CE.DL.C = mean(DL.C.pred.y.tt == Data.tt.Y)
CE.DL.C

