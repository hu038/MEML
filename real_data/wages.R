rm(list=ls())
.libPaths(c("C:/Rlib"))
library(nlme)
library(caret)

library(nlme)
library(tree)
library(randomForest)
library (gbm)
library(e1071)
library(neuralnet)
library(REEMtree)

library(lattice)

load("wages.rda")

wages$hgc=as.numeric(wages$hgc)
wages$race=factor(wages$race)
xyplot(resp~exper|race,data=wages)

#### log response #######################
wages$resp=log(wages$resp)
xyplot(resp~exper|race,data=wages)
xyplot(resp~exper|hgc,data=wages)

#boxplot(resp~exper+race,data=wages,plot=TRUE,col=c("red","blue","green"))

#vecmean <- tapply(wages$resp,c(wages$exper),mean)

wages$id=factor(wages$id)
id_number=as.numeric(dimnames(table(wages$id))[[1]])



fit1=lme(resp~exper+hgc+race,data=wages,random=~1|id,
    method="ML")
fit2=lme(resp~exper+hgc+race,data=wages,random=~race|id,
         method="ML")
fit3=lme(resp~exper+hgc+race,data=wages,random=~hgc|id, 
         method="ML")  
fit4=lme(resp~exper+hgc+race,data=wages,random=~hgc+race|id, 
         method="ML") 
anova(fit1,fit2,fit3,fit4)   ##fit3 has smallest AIC

library(geepack)
library(MESS)
fit4=geeglm(resp~exper+hgc+race,data=wages,id=id,corstr="ar1")
fit5=update(fit4,corstr="exchangeable")
fit6=update(fit4,corstr="unstructured")   ###fit4 has smallest CIC
QIC(fit4)
QIC(fit5)
QIC(fit6)

fit7=REEMtree(resp~exper+hgc+race,data=wages,random=~1|id)
fit8=REEMtree(resp~exper+hgc+race,data=wages,random=~race|id)
fit9=REEMtree(resp~exper+hgc+race,data=wages,random=~hgc|id) 
fit10=REEMtree(resp~exper+hgc+race,data=wages,random=~hgc+race|id)
fit7
fit8
fit9   ## fit9 has largest -2loglikelhood value
fit10

###n=888, eight fold cross-validation
folds <- createFolds(y=id_number,k=8)


###use the training data to predict other subject/id
RMSE.svmk=RMSE.lm=RMSE.lme=RMSE.lmeh=RMSE.geeglm=RMSE.REEM=RMSE.REEMh=RMSE.tree=RMSE.bag=RMSE.rf=RMSE.boost=RMSE.svm=RMSE.nn=rep(0,8)
NMSE.svmk=NMSE.lm=NMSE.lme=NMSE.lmeh=NMSE.geeglm=NMSE.REEM=NMSE.REEMh=NMSE.tree=NMSE.bag=NMSE.rf=NMSE.boost=NMSE.svm=NMSE.nn=rep(0,8)
for(m in 1:8){
 
  traindata=wages[!(wages$id %in% id_number[folds[[m]]]),]
  testdata=wages[wages$id %in% id_number[folds[[m]]],]
  
  fit.lm=lm(resp~exper+hgc+race,data=traindata)
  fit.lme=lme(resp~exper+hgc+race,data=traindata,random=~1|id,
              method="ML")
  fit.lmeh=lme(resp~exper+hgc+race,data=traindata,random=~hgc|id,control = lmeControl(opt='optim'),
               method="ML")
  fit.geeglm=geeglm(resp~exper+hgc+race,data=traindata,id=id,corstr="ar1")
  fit.REEM<-REEMtree(resp~exper+hgc+race,data=traindata,random=~1|id)
  #fit.REEMh<-REEMtree(resp~exper+race,data=traindata,random=~hgc|id)
  fit.tree<-tree::tree(resp~exper+hgc+race,data=traindata)
  fit.bag =randomForest(resp~exper+hgc+race,data=traindata,mtry=3, importance =TRUE)
  fit.rf =randomForest(resp~exper+hgc+race,data=traindata,mtry=1, importance =TRUE)
  fit.boost =gbm(resp~exper+hgc+race,data=traindata, distribution=
                   "gaussian",n.trees =5000,interaction.depth =4)
  fit.svm=svm(resp~exper+hgc+race,data=traindata, kernel="linear")
  fit.svmk=svm(resp~exper+hgc+race,data=traindata, kernel="polynomial")
  nndata=model.matrix(~resp+exper+hgc+race,data=traindata)
  #fit.nn=neuralnet(resp~exper+hgc+racehisp+racewhite,nndata,hidden=3,
  #                 linear.output = TRUE,stepmax=1e6)
  #fit.nn=neuralnet(resp~exper+hgc+racehisp+racewhite,nndata,hidden=3,
  #                 algorithm = "backprop",learningrate=0.0001)
  
  fit.nn=train(resp~exper+hgc+racehisp+racewhite,nndata,method='nnet',linout=TRUE,trace=FALSE)
 
  a.lm=predict(fit.lm,testdata)
  a.lme=predict(fit.lme, testdata,level=0)
  a.lmeh=predict(fit.lmeh, testdata,level=0)
  a.geeglm=predict(fit.geeglm, testdata)
  a.REEM=predict(fit.REEM, testdata,id=testdata$id, EstimateRandomEffects=TRUE)
  #a.REEMh=predict(fit.REEMh, testdata,id=testdata$id, EstimateRandomEffects=TRUE)
  a.tree=predict(fit.tree, testdata)
  a.bag=predict(fit.bag, testdata)
  a.rf=predict(fit.rf, testdata)
  a.boost=predict(fit.boost, testdata,n.trees =5000)
  a.svm=predict(fit.svm, testdata)
  a.svmk=predict(fit.svm, testdata)
  nndatatest=model.matrix(~resp+exper+hgc+race,data=testdata)
  a.nn=predict(fit.nn,nndatatest)
  
  ni=length(testdata$resp)
  
  RMSE.lm[m]=sqrt(sum((testdata$resp-a.lm)^2)/ni)
  RMSE.lme[m]=sqrt(sum((testdata$resp-a.lme)^2)/ni)
  RMSE.lmeh[m]=sqrt(sum((testdata$resp-a.lmeh)^2)/ni)
  RMSE.geeglm[m]=sqrt(sum((testdata$resp-a.geeglm)^2)/ni)
  RMSE.REEM[m]=sqrt(sum((testdata$resp-a.REEM)^2)/ni)
  #RMSE.REEMh[m]=sqrt(sum((testdata$resp-a.REEMh)^2)/ni)
  RMSE.tree[m]=sqrt(sum((testdata$resp-a.tree)^2)/ni)
  RMSE.bag[m]=sqrt(sum((testdata$resp-a.bag)^2)/ni)
  RMSE.rf[m]=sqrt(sum((testdata$resp-a.rf)^2)/ni)
  RMSE.boost[m]=sqrt(sum((testdata$resp-a.boost)^2)/ni)
  RMSE.svm[m]=sqrt(sum((testdata$resp-a.svm)^2)/ni)
  RMSE.svmk[m]=sqrt(sum((testdata$resp-a.svmk)^2)/ni)
  RMSE.nn[m]=sqrt(sum((nndatatest[,2]-a.nn)^2)/ni)
  
  
  MSE=sum((testdata$resp-mean(testdata$resp))^2)
  
 
  NMSE.lm[m]=sum((testdata$resp-a.lm)^2)/MSE
  NMSE.lme[m]=sum((testdata$resp-a.lme)^2)/MSE
  NMSE.lmeh[m]=sum((testdata$resp-a.lmeh)^2)/MSE
  NMSE.geeglm[m]=sum((testdata$resp-a.geeglm)^2)/MSE
  NMSE.REEM[m]=sum((testdata$resp-a.REEM)^2)/MSE
  #NMSE.REEMh[m]=sum((testdata$resp-a.REEMh)^2)/MSE
  NMSE.tree[m]=sum((testdata$resp-a.tree)^2)/MSE
  NMSE.bag[m]=sum((testdata$resp-a.bag)^2)/MSE
  NMSE.rf[m]=sum((testdata$resp-a.rf)^2)/MSE
  NMSE.boost[m]=sum((testdata$resp-a.boost)^2)/MSE
  NMSE.svm[m]=sum((testdata$resp-a.svm)^2)/MSE
  NMSE.svmk[m]=sum((testdata$resp-a.svmk)^2)/MSE
  NMSE.nn[m]=sum((nndatatest[,2]-a.nn)^2)/MSE
}


Aresult_NMSE=cbind(NMSE.lm,NMSE.lme,NMSE.lmeh,NMSE.geeglm,NMSE.REEM,
                      NMSE.tree, NMSE.bag,NMSE.rf,NMSE.boost,NMSE.svm,NMSE.svmk,NMSE.nn)

Aresult_RMSE=cbind(RMSE.lm,RMSE.lme,RMSE.lmeh,RMSE.geeglm,RMSE.REEM,
                   RMSE.tree, RMSE.bag,RMSE.rf,RMSE.boost,RMSE.svm,RMSE.svmk,RMSE.nn)



NMSE_log_mean=apply(Aresult_NMSE,2,mean)
RMSE_log_mean=apply(Aresult_RMSE,2,mean)

#### remove lm, lmeh, geglm
## plot RMSE
RMSE_log_mean=RMSE_log_mean[-c(1,3,4)]
M=c("lme", "re-em tree","tree","bag","rf","boost","svm","svmk","nn")
barplot(RMSE_log_mean,names.arg = M,xlab="Method",ylab="RMSE value",col="blue",
        border="red")


boxplot(Aresult_RMSE)

### load("wages.RData")

knitr::kable(round(t(RMSE_log_mean),3),format="latex")
