##This is the simulation code for the comparing statistical model and machine learning methods in longitudinal data

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

##################   AR(1) correlation structure
ar1 <- function(r,n)      
{
  lags <- diag(n)
  lags<-abs(row(lags)-col(lags))
  r^lags
}

########## correlated data generation
# y:  normal;
# x_i: uniform+c(1:n); random x.
# err_i: norm

gennorm<-function(K,n,b,cv) 
{
  y<-matrix(NA,K*n,7)
  dimnames(y)<-list(NULL,c("id","yi","xi","xi2","mu","time","err"))
  for (i in 1:K) {
    xi<- runif(n,-1,1)
    #+c(1,n) 
    id<-rep(i,n)
    time=1:n
    xi2=rbinom(1,1,0.5)
    mu<-b[1]+b[2]*xi+b[3]*xi2+b[4]*xi^2
    err<-t(cv)%*%rnorm(n)         #cv<-chol(cv), upper triangula	
    yi<-mu+sigma*err
    y[((i-1)*n+1):(i*n),] <-cbind(id,yi,xi,xi2,mu,time,err)
  }
  y
} 
####generate norm data
K=100
n=5
rho=0.5  ##ar1 data
sigma=1
btrue=c(0,0.5,1,-5)



cv=chol(ar1(rho,n))  # true AR1

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5)
tuneGridboost <- expand.grid(
  n.trees = c(500, 1000, 2000, 5000),
  interaction.depth = c(1, 2),
  shrinkage = 0.1,
  n.minobsinnode = 10)


sim=1000

####################two-step prediction
##n=5

RMSE.merf=RMSE.nntanh=RMSE.svmkernel=RMSE.lm=RMSE.lme=RMSE.REEM=RMSE.tree=RMSE.bag=RMSE.rf=RMSE.boost=RMSE.svm=RMSE.nn=matrix(0,sim,3)
RMSE1.merf=RMSE1.nntanh=RMSE1.svmkernel=RMSE1.lm=RMSE1.lme=RMSE1.REEM=RMSE1.tree=RMSE1.bag=RMSE1.rf=RMSE1.boost=RMSE1.svm=RMSE1.nn=matrix(0,sim,3)

beta1T=matrix(0,sim,3)
beta2T=matrix(0,sim,3)

for(s in 1:sim)
{
  cat("sim = ", s, "\n")
  data=as.data.frame(gennorm(K,n,btrue,cv))
 
  for(j in 3:5){
  subset1= (data$time %in% j) 
  test=data[subset1,]
  subset2= (data$time %in% c(1:j-2)) 
  train=data[subset2,]
  
  fit.merf <- MERF(X=cbind(train$xi, train$xi2),Y=train$yi,Z=as.matrix(rep(1,nrow(train))),id=train$id,
                   time=train$time,mtry=1,ntree=500,sto="none") 
 fit.lm=lm(yi~xi+xi2,train)
  fit.lme=lme(yi~xi+xi2,train,random=~1|id,
              method="ML")
  fit.REEM<-REEMtree::REEMtree(yi~xi+xi2,train,random=~1|id)
  fit.tree <-caret::train(yi~xi+xi2,
                          data = train,
                          method = "rpart2",
                          tuneLength=10,
                          trControl = ctrl)
  fit.boost <-caret::train(yi~xi+xi2,
                           data = train,
                           method = "gbm",
                           verbose = FALSE,
                           trControl = ctrl,
                           tuneGrid = tuneGridboost)
  #fit.tree=tree::tree(yi~xi+xi2,train)
  fit.bag =randomForest(yi~xi+xi2,train,mtry=2, importance =TRUE)
  fit.rf =randomForest(yi~xi+xi2,train,mtry=1, importance =TRUE)
  #fit.boost =gbm(yi~xi+xi2,train, distribution=
  #  "gaussian",n.trees =2000,interaction.depth =4)
  fit.svm=svm(yi~xi+xi2,train, kernel="linear")
  fit.svmkernel=svm(yi~xi+xi2,train, kernel="polynomial",degree=2)
  fit.nn=nn.train(x=cbind(train$xi, train$xi2),y=train$yi, hidden = c(2), output="linear")
  fit.nntanh=nn.train(x=cbind(train$xi, train$xi2),y=train$yi, hidden = c(2), activationfun="tanh", output="linear")
  
  beta1T[s,j-2]=fit.lm$coefficients[2]
  beta2T[s,j-2]=fit.lm$coefficients[3]
  
  a.lm=predict(fit.lm,test)
  a.lme=predict(fit.lme,test)
  a.REEM=predict(fit.REEM, test,id=test$id, EstimateRandomEffects=TRUE)
  a.tree=predict(fit.tree, test)
  a.bag=predict(fit.bag, test)
  a.rf=predict(fit.rf, test)
  a.boost=predict(fit.boost, test,n.trees =fit.boost$bestTune$n.trees)
  a.svm=predict(fit.svm, test)
  a.svmkernel=predict(fit.svmkernel, test)
  a.nn=nn.predict(fit.nn,cbind(test$xi, test$xi2))
  a.nntanh=nn.predict(fit.nntanh,cbind(test$xi, test$xi2))
  a.merf=predict(fit.merf,X=cbind(test$xi, test$xi2),Z=as.matrix(rep(1,nrow(test)))
                 ,id=test$id, time=test$time)
  
  ni=length(test$yi)
  
  RMSE1.lm[s,j-2]=sqrt(sum((test$mu-a.lm)^2)/ni)
  RMSE1.lme[s,j-2]=sqrt(sum((test$mu-a.lme)^2)/ni)
  RMSE1.REEM[s,j-2]=sqrt(sum((test$mu-a.REEM)^2)/ni)
  RMSE1.tree[s,j-2]=sqrt(sum((test$mu-a.tree)^2)/ni)
  RMSE1.bag[s,j-2]=sqrt(sum((test$mu-a.bag)^2)/ni)
  RMSE1.rf[s,j-2]=sqrt(sum((test$mu-a.rf)^2)/ni)
  RMSE1.boost[s,j-2]=sqrt(sum((test$mu-a.boost)^2)/ni)
  RMSE1.svm[s,j-2]=sqrt(sum((test$mu-a.svm)^2)/ni)
  RMSE1.svmkernel[s,j-2]=sqrt(sum((test$mu-a.svmkernel)^2)/ni)
  RMSE1.nn[s,j-2]=sqrt(sum((test$mu-a.nn)^2)/ni)
  RMSE1.nntanh[s,j-2]=sqrt(sum((test$mu-a.nntanh)^2)/ni)
  RMSE1.merf[s,j-2]=sqrt(sum((test$mu-a.merf)^2)/ni)
  
  RMSE.lm[s,j-2]=sqrt(sum((test$yi-a.lm)^2)/ni)
  RMSE.lme[s,j-2]=sqrt(sum((test$yi-a.lme)^2)/ni)
  RMSE.REEM[s,j-2]=sqrt(sum((test$yi-a.REEM)^2)/ni)
  RMSE.tree[s,j-2]=sqrt(sum((test$yi-a.tree)^2)/ni)
  RMSE.bag[s,j-2]=sqrt(sum((test$yi-a.bag)^2)/ni)
  RMSE.rf[s,j-2]=sqrt(sum((test$yi-a.rf)^2)/ni)
  RMSE.boost[s,j-2]=sqrt(sum((test$yi-a.boost)^2)/ni)
  RMSE.svm[s,j-2]=sqrt(sum((test$yi-a.svm)^2)/ni)
  RMSE.svmkernel[s,j-2]=sqrt(sum((test$yi-a.svmkernel)^2)/ni)
  RMSE.nn[s,j-2]=sqrt(sum((test$yi-a.nn)^2)/ni)
  RMSE.nntanh[s,j-2]=sqrt(sum((test$yi-a.nntanh)^2)/ni)
  RMSE.merf[s,j-2]=sqrt(sum((test$yi-a.merf)^2)/ni)
 }

}

save.image("data_p2_ar1_100_5_W.RData")







