# Benchmark codes (Jinsung Yoon, 02/08/2017)
# For Classification Problem
rm(list = ls())

# Necessary Pakages
# 1. Cross-Validation
#install.packages('cvTools')
# 2. Linear/Logistic Regression
#install.packages("Rcpp")
# 3. Random Forest
#install.packages("randomForest")
# 4. LASSO
#install.packages("glmnet")
# 5. Decision Tree
#install.packages("party")
# 6. Boosting (AdaBoost & LogitBoost)
#install.packages("mboost")
#install.packages("gbm")
# 7. Neural Nets
#install.packages("neuralnet")
# 8. XGBoost
#install.packages("xgboost")
#install.packages("readr")
#install.packages("stringr")
#install.packages("caret")
#install.packages("car")
# 9. Deep Boost
#install.packages("deepboost")

# 0. AUC computation
#install.packages("pROC")

# Libraries based on the packages
library("cvTools")
library("Rcpp")
library("randomForest")
library("glmnet")
library("party")
library("mboost")
library("gbm")
library("neuralnet")
library("xgboost")
library("readr")
library("stringr")
library("caret")
library("car")
library("deepboost")

library("pROC")

# System Parameters
# 1. # of Cross-validation
KFold <- 10
# 2. # of Algorithms
Algo_No <- 7
# 3. # of Trees
Tree_No <- 100
# 4. Neural Nets
NN_No = 10
NN_Depth = 10


# Data Input
#Data <- read.csv('C:/Users/Jinsung/Desktop/RCodes/Data/BreastCancerOriginal.csv')
Data <- read.csv('C:/Users/Jinsung/Desktop/RCodes/Data/mnist_train49.csv')
Feature_No = ncol(Data)-1

# Cross Validation
# Divide the folds
folds <- cvFolds(nrow(Data), K = KFold)

# Outputs
Score = matrix(0,nrow(Data),ncol=Algo_No)
AUC <- matrix(0,nrow = KFold,ncol = Algo_No)
Final_AUC <- matrix(0,nrow = Algo_No,ncol = 1)
New_Score = matrix(0,nrow = nrow(Score),ncol = ncol(Score))


# Cross Validation Iteration
for (iter in 1:KFold) {
  # Train/Test Index  
  train_idx <- folds$subsets[which(folds$which != iter)]
  test_idx <- folds$subsets[which(folds$which == iter)]
  
  # Train/Test Assign
  Train = Data[train_idx,]
  Test = Data[test_idx,]
  
  Train_Feature = do.call(cbind,Data[train_idx,1:Feature_No],1)
  Test_Feature = do.call(cbind,Data[test_idx,1:Feature_No],1)
  
  # 1. Linear Regression
  LinearR <- lm(Label~.,data = Train)
  Score[test_idx,1] <- predict(LinearR,newdata = Test)
  
  # 2. Logistic Regression
  LogitR <- glm(Label ~., data = Train, family = "binomial")
  Score[test_idx,2] <- predict(LogitR, Test, type = "response")
  
  # 3. Random Forest
  RForest <- randomForest(Label~., data = Train, ntree = Tree_No)
  Score[test_idx,3] <- predict(RForest,newdata=Test)
  
  # 4. LASSO
  LASSO <- cv.glmnet(x = Train_Feature, y = Train$Label, family="binomial", type.measure = "mse", nfolds = 20)
  Score[test_idx,4] <- predict(LASSO, newx = Test_Feature, type = "response")
  
  # 5. Decision Tree
  TreeR <- ctree(Label~. , data=Train)
  Score[test_idx,5] <- predict(TreeR,Test)
  
  # 6. AdaBoost
  #AdaR <- gbm(Label~., data=Train, distribution = "adaboost", n.trees = 5000)
  #Score[,6] <- predict(AdaR,Test,n.trees = 5000, type = 'response')
  
  # 7. LogitBoost
  LogitR <- glmboost(Label~., data=Train)
  Score[test_idx,6] <- predict(LogitR,Test)
  
  # 8. Neural Networks
  #name_vec <- names(Train)
  #f_name <- as.formula(paste("Label ~", paste(name_vec[!name_vec %in% "Label"], collapse = " + ")))
  #nnR <- neuralnet(f_name,data=Train,hidden=c(NN_No,NN_Depth),linear.output=T)
  #Score[test_idx,8] <- compute(nnR,Test[,1:Feature_No])$net.result

  # 9. xGBoost  
  for (i in 1:nrow(Train_Feature)){
    for (j in 1:ncol(Train_Feature)){
      Train_Feature[i,j] <- as.numeric(Train_Feature[i,j])
    }
  }
  xgb <- xgboost(data = Train_Feature, label = Train$Label,nrounds = Tree_No, objective = "binary:logistic")
  
  for (i in 1:nrow(Test_Feature)){
    for (j in 1:ncol(Test_Feature)){
      Test_Feature[i,j] <- as.numeric(Test_Feature[i,j])
    }
  }
  Score[test_idx,7] <- predict(xgb,Test_Feature)
  
  # 10. DeepBoost
  #best_params <- deepboost.gridSearch(Label~., data = Train)
  #Deep <- deepboost(Label~., data = Train,num_iter = best_params[2][[1]], 
  #beta = best_params[3][[1]], 
  #lambda = best_params[4][[1]], 
  #loss_type = best_params[5][[1]]
  #)
#  Deep <- deepboost(Label~., data = Train, num_iter = Tree_No)
 # Score[test_idx,10] <- as.numeric(deepboost.predict(Deep,Test))
  
  
  # AUC Computation
  
  for (i in 1:nrow(Score)){
    for (j in 1:ncol(Score)){
      New_Score[i,j] <- as.numeric(Score[i,j])
    }
  }
  
  for (i in 1:Algo_No){
    AUC[iter,i] <- auc(Test$Label,New_Score[test_idx,i])
  }
  
  
}


# AUC computation
for (i in 1:nrow(Score)){
  for (j in 1:ncol(Score)){
    New_Score[i,j] <- as.numeric(Score[i,j])
  }
}

for (i in 1:Algo_No){
  Final_AUC[i] <- auc(Data$Label,New_Score[,i])
}


