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
library(deepnet)
library(caret)

K=100
K1=50
n=10

btrue=c(0.5,1,1.2,-5)
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
  
  mu=btrue[1] + btrue[2]*x + btrue[3]*t+ btrue[4]*x^2
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

#################### new objects ########################3
##K=100, predict other 50 subjects

RMSE1.merf=RMSE1.nntanh=RMSE1.svmkernel=RMSE1.lm=RMSE1.lme=RMSE1.REEM=RMSE1.tree=RMSE1.bag=RMSE1.rf=RMSE1.boost=RMSE1.svm=RMSE1.nn=rep(0,sim)
RMSE.merf=RMSE.nntanh=RMSE.svmkernel=RMSE.lm=RMSE.lme=RMSE.REEM=RMSE.tree=RMSE.bag=RMSE.rf=RMSE.boost=RMSE.svm=RMSE.nn=rep(0,sim)
BIAS.merf=BIAS.nntanh=BIAS.svmkernel=BIAS.lm=BIAS.lme=BIAS.REEM=BIAS.tree=BIAS.bag=BIAS.rf=BIAS.boost=BIAS.svm=BIAS.nn=rep(0,sim)


for(s in 1:sim)
{
  cat("sim = ", s, "\n")
  data=genlme(K,n,btrue,sds,sd)
  
  train=data
  
  test=genlme(K1,n,btrue,sds,sd)
  
  fit.merf <- MERF(X=cbind(train$x, train$t),Y=train$y,Z=as.matrix(rep(1,nrow(train))),id=train$id,
                   time=train$time,mtry=1,ntree=500,sto="none")
  fit.lm=lm(y~x+t,train)
  fit.lme=lme(y~x+t,train,random=~1|id,
              method="ML")
  fit.REEM<-REEMtree::REEMtree(y~x+t,train,random=~1|id)
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
  #fit.tree=tree::tree(y~x+t,train)
  fit.bag =randomForest(y~x+t,train,mtry=2, importance =TRUE)
  fit.rf =randomForest(y~x+t,train,mtry=1, importance =TRUE)
  #fit.boost =gbm(y~x+t,train, distribution=
   #                "gaussian",n.trees =5000,interaction.depth =4)
  fit.svm=svm(y~x+t,train, kernel="linear", scale=FALSE)
  fit.svmkernel=svm(y~x+t,train, kernel="polynomial")
  fit.nn=nn.train(x=cbind(train$x, train$t),y=train$y, hidden = c(2), output="linear")
  fit.nntanh=nn.train(x=cbind(train$x, train$t),y=train$y, hidden = c(2), activationfun="tanh", output="linear")
  
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
  RMSE.merf[s]=sqrt(sum((test$y-a.merf)^2)/ni)
  RMSE.lm[s]=sqrt(sum((test$y-a.lm)^2)/ni)
  RMSE.lme[s]=sqrt(sum((test$y-a.lme)^2)/ni)
  RMSE.REEM[s]=sqrt(sum((test$y-a.REEM)^2)/ni)
  RMSE.tree[s]=sqrt(sum((test$y-a.tree)^2)/ni)
  RMSE.bag[s]=sqrt(sum((test$y-a.bag)^2)/ni)
  RMSE.rf[s]=sqrt(sum((test$y-a.rf)^2)/ni)
  RMSE.boost[s]=sqrt(sum((test$y-a.boost)^2)/ni)
  RMSE.svm[s]=sqrt(sum((test$y-a.svm)^2)/ni)
  RMSE.svmkernel[s]=sqrt(sum((test$y-a.svmkernel)^2)/ni)
  RMSE.nn[s]=sqrt(sum((test$y-a.nn)^2)/ni)
  RMSE.nntanh[s]=sqrt(sum((test$y-a.nntanh)^2)/ni)
  
  BIAS.merf[s]=sum(a.merf-test$y)/ni
  BIAS.lm[s]=sum(a.lm-test$y)/ni
  BIAS.lme[s]=sum(a.lme-test$y)/ni
  BIAS.REEM[s]=sum(a.REEM-test$y)/ni
  BIAS.tree[s]=sum(a.tree-test$y)/ni
  BIAS.bag[s]=sum(a.bag-test$y)/ni
  BIAS.rf[s]=sum(a.rf-test$y)/ni
  BIAS.boost[s]=sum(a.boost-test$y)/ni
  BIAS.svm[s]=sum(a.svm-test$y)/ni
  BIAS.svmkernel[s]=sum(a.svmkernel-test$y)/ni
  BIAS.nn[s]=sum(a.nn-test$y)/ni
  BIAS.nntanh[s]=sum(a.nntanh-test$y)/ni
  
  RMSE1.merf[s]=sqrt(sum((test$mu-a.merf)^2)/ni)
  RMSE1.lm[s]=sqrt(sum((test$mu-a.lm)^2)/ni)
  RMSE1.lme[s]=sqrt(sum((test$mu-a.lme)^2)/ni)
  RMSE1.REEM[s]=sqrt(sum((test$mu-a.REEM)^2)/ni)
  RMSE1.tree[s]=sqrt(sum((test$mu-a.tree)^2)/ni)
  RMSE1.bag[s]=sqrt(sum((test$mu-a.bag)^2)/ni)
  RMSE1.rf[s]=sqrt(sum((test$mu-a.rf)^2)/ni)
  RMSE1.boost[s]=sqrt(sum((test$mu-a.boost)^2)/ni)
  RMSE1.svm[s]=sqrt(sum((test$mu-a.svm)^2)/ni)
  RMSE1.svmkernel[s]=sqrt(sum((test$mu-a.svmkernel)^2)/ni)
  RMSE1.nn[s]=sqrt(sum((test$mu-a.nn)^2)/ni)
  RMSE1.nntanh[s]=sqrt(sum((test$mu-a.nntanh)^2)/ni)
  
  
}

save.image("data_no_W.RData")







