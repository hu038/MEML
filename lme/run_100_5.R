##This is the simulation code for the comparing statistical model 
#and machine learning methods in longitudinal data

## generate data from mixed model
rm(list=ls())
library(nlme)
library(tree)
library(randomForest)
library (gbm)
library(e1071)
library(neuralnet)
library(knitr)
library(REEMtree)
library(LongituRF)
library(caret)
library(deepnet)

K=100
K1=50
n=5

btrue=c(0.5,1,1.2)
sds = 2
sd = 1

genlme=function(K,n,btrue,sds,sd){
  id = rep(1:K, each = n)
  #id=as.factor(id)
  time=rep(1:n, K)
  
  Keff = rep( rnorm(K, 0, sds), each = n)
  neff = rnorm(K*n, 0, sd)
  
  x = rep( runif(K, 0, 1), each = n)
  
  t = runif(K*n, 0, 1) 
  
  mu=btrue[1] + btrue[2]*x + btrue[3]*t
  y = mu + Keff + neff
  
  dat=data.frame(id,x,t,y,time,mu)
  dat
}

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5)
tuneGridboost <- expand.grid(
  n.trees = c(500, 1000, 2000, 5000),
  interaction.depth = c(1, 2),
  shrinkage = 0.1,
  n.minobsinnode = 10)

sim=1000

####################one-step prediction
##n=5

RMSE.merf=RMSE.nntanh=RMSE.svmkernel=RMSE.lm=RMSE.lme=RMSE.REEM=RMSE.tree=RMSE.bag=RMSE.rf=RMSE.boost=RMSE.svm=RMSE.nn=matrix(0,sim,4)
RMSE1.merf=RMSE1.nntanh=RMSE1.svmkernel=RMSE1.lm=RMSE1.lme=RMSE1.REEM=RMSE1.tree=RMSE1.bag=RMSE1.rf=RMSE1.boost=RMSE1.svm=RMSE1.nn=matrix(0,sim,4)

beta1T=matrix(0,sim,4)
beta2T=matrix(0,sim,4)
for(s in 1:sim)
{
  cat("sim = ", s, "\n")
  data=genlme(K,n,btrue,sds,sd)
 
  for(j in 2:5){
  subset1= (data$time %in% j) 
  test=data[subset1,]
  subset2= (data$time %in% c(1:j-1)) 
  train=data[subset2,]
  
  fit.merf <- MERF(X=cbind(train$x, train$t),Y=train$y,Z=as.matrix(rep(1,nrow(train))),id=train$id,
                   time=train$time,mtry=1,ntree=500,sto="none")
  fit.lm=lm(y~x+t,train)
  fit.lme=lme(y~x+t,train,random=~1|id,
              method="ML")
  fit.REEM<-REEMtree::REEMtree(y~x+t,train,random=~1|id)
  #fit.tree=tree::tree(y~x+t,train)
  fit.tree <-caret::train(y~x+t,
                          data = train,
                          method = "rpart2",
                          tuneLength=10,
                          trControl = ctrl)
  fit.boost <-caret::train(y~x+t,
                           data = train,
                           method = "gbm",
                           verbose = FALSE,
                           trControl = ctrl,
                           tuneGrid = tuneGridboost)
  fit.bag =randomForest(y~x+t,train,mtry=2, importance =TRUE)
  fit.rf =randomForest(y~x+t,train,mtry=1, importance =TRUE)
  #fit.boost =gbm(y~x+t,train, distribution=
  #                 "gaussian",n.trees =5000,interaction.depth =4)
  fit.svm=svm(y~x+t,train, kernel="linear", scale=FALSE)
  fit.svmkernel=svm(y~x+t,train, kernel="polynomial")
  fit.nn=nn.train(x=cbind(train$x, train$t),y=train$y, hidden = c(2), output="linear")
  fit.nntanh=nn.train(x=cbind(train$x, train$t),y=train$y, hidden = c(2), activationfun="tanh", output="linear")
  
  beta1T[s,j-1]=fit.lm$coefficients[2]
  beta2T[s,j-1]=fit.lm$coefficients[3]
  
  a.merf=predict(fit.merf,X=cbind(test$x, test$t),Z=as.matrix(rep(1,nrow(test)))
                 ,id=test$id, time=test$time)
  a.lm=predict(fit.lm,test)
  a.lme=predict(fit.lme,test)
  a.REEM=predict(fit.REEM, test,id=test$id, EstimateRandomEffects=TRUE)
  a.tree=predict(fit.tree, test)
  a.bag=predict(fit.bag, test)
  a.rf=predict(fit.rf, test)
  a.boost=predict(fit.boost, test,n.trees =fit.boost$bestTune$n.trees)
  a.svm=predict(fit.svm, test)
  a.svmkernel=predict(fit.svmkernel, test)
  a.nn=nn.predict(fit.nn,cbind(test$x, test$t))
  a.nntanh=nn.predict(fit.nntanh,cbind(test$x, test$t))
  
  ni=length(test$y)
  RMSE1.merf[s,j-1]=sqrt(sum((test$mu-a.merf)^2)/ni)
  RMSE1.lm[s,j-1]=sqrt(sum((test$mu-a.lm)^2)/ni)
  RMSE1.lme[s,j-1]=sqrt(sum((test$mu-a.lme)^2)/ni)
  RMSE1.REEM[s,j-1]=sqrt(sum((test$mu-a.REEM)^2)/ni)
  RMSE1.tree[s,j-1]=sqrt(sum((test$mu-a.tree)^2)/ni)
  RMSE1.bag[s,j-1]=sqrt(sum((test$mu-a.bag)^2)/ni)
  RMSE1.rf[s,j-1]=sqrt(sum((test$mu-a.rf)^2)/ni)
  RMSE1.boost[s,j-1]=sqrt(sum((test$mu-a.boost)^2)/ni)
  RMSE1.svm[s,j-1]=sqrt(sum((test$mu-a.svm)^2)/ni)
  RMSE1.svmkernel[s,j-1]=sqrt(sum((test$mu-a.svmkernel)^2)/ni)
  RMSE1.nn[s,j-1]=sqrt(sum((test$mu-a.nn)^2)/ni)
  RMSE1.nntanh[s,j-1]=sqrt(sum((test$mu-a.nntanh)^2)/ni)
  
  RMSE.merf[s,j-1]=sqrt(sum((test$y-a.merf)^2)/ni)
  RMSE.lm[s,j-1]=sqrt(sum((test$y-a.lm)^2)/ni)
  RMSE.lme[s,j-1]=sqrt(sum((test$y-a.lme)^2)/ni)
  RMSE.REEM[s,j-1]=sqrt(sum((test$y-a.REEM)^2)/ni)
  RMSE.tree[s,j-1]=sqrt(sum((test$y-a.tree)^2)/ni)
  RMSE.bag[s,j-1]=sqrt(sum((test$y-a.bag)^2)/ni)
  RMSE.rf[s,j-1]=sqrt(sum((test$y-a.rf)^2)/ni)
  RMSE.boost[s,j-1]=sqrt(sum((test$y-a.boost)^2)/ni)
  RMSE.svm[s,j-1]=sqrt(sum((test$y-a.svm)^2)/ni)
  RMSE.svmkernel[s,j-1]=sqrt(sum((test$y-a.svmkernel)^2)/ni)
  RMSE.nn[s,j-1]=sqrt(sum((test$y-a.nn)^2)/ni)
  RMSE.nntanh[s,j-1]=sqrt(sum((test$y-a.nntanh)^2)/ni)
 }

  
}

save.image("data_100_5.RData")







