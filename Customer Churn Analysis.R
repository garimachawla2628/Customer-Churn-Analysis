#-------------------loading packages---------------------------#

library(plyr)        # for mapvalues()
library(rpart.plot) 
library(caret)    # for createDataPartition 
library(gridExtra)   # for arranging plots in grid
library(e1071)       # for naive bayes
library(ggplot2)
library(dplyr)             # for glimplse() 
library(ROCR)      
library(pROC)     # for roc curve
library(rpart)
library(randomForest)
library(neuralnet)
library(keras)       # for neural plot


#-------------Telecom Customer Churn Analysis-----------------#

#1. Load & understand data

churn<-read.csv(file.choose(),header=TRUE)
View(churn)
glimpse(churn)

#2 Data pre-processing

#2.1 Handling missing values

colSums(is.na(churn))
NAIndex <- which(is.na(churn$TotalCharges))
churn <- churn[-c(NAIndex),]
churn <- na.omit(churn)

#2.2 Removing unwanted columns

churn <- churn[-1]

#2.3 Improving data readability

churn$SeniorCitizen <- as.factor(mapvalues(churn$SeniorCitizen,
                                                 from=c("0","1"),
                                                 to=c("No", "Yes")))

#2.4 Removing data redundancy

churn$MultipleLines <- as.factor(mapvalues(churn$MultipleLines, 
                                                 from=c("No phone service"),
                                                 to=c("No")))

for(i in 10:15){
  churn[,i] <- as.factor(mapvalues(churn[,i],
                                   from= c("No internet service"), 
                                   to= c("No")))
}

#2.5 Improvising data types

str(churn)
churn$gender <- as.factor(churn$gender)
churn$Partner <- as.factor(churn$Partner)
churn$Dependents <- as.factor(churn$Dependents)
churn$InternetService <- as.factor(churn$InternetService)
churn$PhoneService <- as.factor(churn$PhoneService)
churn$PaperlessBilling <- as.factor(churn$PaperlessBilling)
churn$Churn <- as.factor(churn$Churn)
churn$Contract <- as.factor(churn$Contract)
churn$PaymentMethod <- as.factor(churn$PaymentMethod)
str(churn)

#3 Data Visualization to understand Demographics of our Customers

#Gender plot
p1 <- ggplot(churn, aes(x = gender)) +
  geom_bar(aes(fill = Churn)) +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) 

#Senior citizen plot
p2 <- ggplot(churn, aes(x = SeniorCitizen)) +
  geom_bar(aes(fill = Churn)) +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Partner plot
p3 <- ggplot(churn, aes(x = Partner)) +
  geom_bar(aes(fill = Churn)) +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Dependents plot
p4 <- ggplot(churn, aes(x = Dependents)) +
  geom_bar(aes(fill = Churn)) +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Plot demographic data within a grid
grid.arrange(p1, p2, p3, p4, ncol=2)

#4 Check Multicollinearity

cor(churn$MonthlyCharges,churn$TotalCharges)

#High correlation (>60%), we can remove TotalCharges

churn$TotalCharges<-NULL
View(churn)

#5 Divide dataset into training & testing set

split_train_test <- createDataPartition(churn$Churn,p=0.7,list=FALSE)

# createDataPartition is a function used from caret package
# y is a factor variable which tries to maintain the partition between the factor levels
# p=0.7, 70% to training data
# The list = FALSE avoids returning the data as a list

dtrain<- churn[split_train_test,]
dtest<-  churn[-split_train_test,]
dim(dtrain)         
dim(dtest)

#6 Modelling

#6.1 Decision Trees

#6.1.1 Model development
set.seed(222)
dt1 <- rpart(Churn ~., data = dtrain, method="class")
rpart.plot(dt1)

# The contract variable is the most important. 
# Customers with one-year contracts are more likely to 
# churn. Customers with DSL internet service are less likely 
# to churn. Customers who have stayed longer than 15 months 
# are less likely to churn. 

#6.1.2 Prediction & Confusion Matrix

tr_prob1 <- predict(dt1, dtest)
tr_pred1 <- ifelse(tr_prob1[,2] > 0.5,"Yes","No")
dt1_cm <- table(Predicted = tr_pred1, Actual = dtest$Churn)
dt1_cm

#6.1.3 Accuracy

dt1_acc <- sum(diag(dt1_cm)/nrow(dtest)) * 100
dt1_acc

# The decision tree model is fairly accurate, correctly predicting 
# the churn status of customers in the test subset 79% of the time.

#6.2 Logistic Regression

#6.2.1 Model development

set.seed(333)
lr1 <- glm(as.factor(Churn) ~., data = dtrain, family="binomial")
summary(lr1)

# By examining the significance values, we see similar predictor 
# variables of importance. Tenure length, contract status, and 
# PaperlessBilling have the lowest p-values and can be identified as 
# the best predictors of customer churn.

lr2<-glm(as.factor(Churn)~tenure+Contract+PaperlessBilling,data=dtrain,family="binomial")
summary(lr2)

#6.2.2 Prediction & Confusion Matrix

# For LR1
lr_prob1 <- predict(lr1, dtest, type="response")
lr_pred1 <- ifelse(lr_prob1 > 0.5,"Yes","No")
lr1_cm <- table(Predicted = lr_pred1, Actual = dtest$Churn)
lr1_cm

# For LR2
lr_prob2 <- predict(lr2, dtest, type="response")
lr_pred2 <- ifelse(lr_prob2 > 0.5,"Yes","No")
lr2_cm <- table(Predicted = lr_pred2, Actual = dtest$Churn)
lr2_cm

#6.2.3 Accuracy

# For LR1
lr1_acc <- sum(diag(lr1_cm)/nrow(dtest)) * 100
lr1_acc

# For LR2
lr2_acc <- sum(diag(lr2_cm)/nrow(dtest)) * 100
lr2_acc

# LR1 is a more accurate model than LR2

#6.3 Random Forest

#6.2.1 Model development
set.seed(444)
rfm1 <- randomForest(Churn ~., data = dtrain)
rfm1


# Prediction & Confusion Matrix

rf_pred1 <- predict(rfm1, dtest)
rf_cm1 <- table(Predicted = rf_pred1, Actual = dtest$Churn)
rf_cm1

# Accuracy

rf_acc1 <- sum(diag(rf_cm1)/nrow(dtest)) * 100
rf_acc1

# The current error rate is 20.98% 
# Reducing the number of mtry to 2 and ntree from default of 
# 500 to 200 to check the accuracy
# ntree= no. of trees to produce
# mtry= variables/features to choose per level

rfm2 <- randomForest(Churn ~., data = dtrain,
                     ntree = 200, mtry = 2, 
                     importance = TRUE, proximity = TRUE)
rfm2

# Prediction & Confusion Matrix

rf_pred2 <- predict(rfm2, dtest)
rf_cm2 <- table(Predicted = rf_pred2, Actual = dtest$Churn)

# Accuracy

rf_acc2 <- sum(diag(rf_cm2)/nrow(dtest)) * 100
rf_acc2

# Random Forest 2 Model is better than Random Forest 1

# Most Important Variables

importance_matrix <- data.frame(Variables = 
                                  rownames(rfm2$importance), rfm2$importance, row.names = NULL)
ggplot(data = importance_matrix , 
       aes(y = MeanDecreaseGini , x = Variables, fill = Variables))+ geom_col() + coord_flip() + labs(title= 'Variiable importance plot')+ theme_classic()


# contract, tenure, monthly charges, imp variables


#6.4 Naive Bayes

#6.4.1 Model development
set.seed(555)
nbm<-naiveBayes(Churn~.,data=dtrain)
nbm


#it creates conditional probability table for each of the 
# features and also displays the prior probability 

#6.4.2 Prediction & Confusion Matrix

nbm_pred <- predict(nbm, dtest, type = "class")
nbm_cm <- table(nbm_pred, dtest$Churn)
nbm_cm

#6.4.3 Accuracy

nbm_acc <- sum(diag(nbm_cm)/nrow(dtest))*100
nbm_acc

#6.5 Support Vector Machine

#6.5.1 Model development
set.seed(666)
svm <- svm(Churn~., dtrain, type = "C-classification", kernel = "linear")


#6.5.2 Prediction & Confusion Matrix

svm_pred <- predict(svm, dtest)
svm_cm <- table(svm_pred, dtest$Churn)

#6.5.3 Accuracy

svm_acc <- sum(diag(svm_cm)/nrow(dtest))*100
svm_acc

#6.6 Artificial Neural Networks

#6.6.1 Min Max Normalization
# This algo requires numeric values between 0 and 1
# converting all variables to numeric 
# then normalizing them

churn1 <- churn
churn1 <- na.omit(churn1)

churn1$gender <- (as.numeric(churn1$gender) - min(as.numeric(churn1$gender)))/(max(as.numeric(churn1$gender)) - min(as.numeric(churn1$gender)))
churn1$SeniorCitizen <- (as.numeric(churn1$SeniorCitizen) - min(as.numeric(churn1$SeniorCitizen)))/(max(as.numeric(churn1$SeniorCitizen)) - min(as.numeric(churn1$SeniorCitizen)))
churn1$Partner <- (as.numeric(churn1$Partner) - min(as.numeric(churn1$Partner)))/(max(as.numeric(churn1$Partner)) - min(as.numeric(churn1$Partner)))
churn1$Dependents <- (as.numeric(churn1$Dependents) - min(as.numeric(churn1$Dependents)))/(max(as.numeric(churn1$Dependents)) - min(as.numeric(churn1$Dependents)))
churn1$tenure <- (as.numeric(churn1$tenure) - min(as.numeric(churn1$tenure)))/(max(as.numeric(churn1$tenure)) - min(as.numeric(churn1$tenure)))
churn1$PhoneService <- (as.numeric(churn1$PhoneService) - min(as.numeric(churn1$PhoneService)))/(max(as.numeric(churn1$PhoneService)) - min(as.numeric(churn1$PhoneService)))
churn1$MultipleLines <- (as.numeric(churn1$MultipleLines) - min(as.numeric(churn1$MultipleLines)))/(max(as.numeric(churn1$MultipleLines)) - min(as.numeric(churn1$MultipleLines)))
churn1$InternetService <- (as.numeric(churn1$InternetService) - min(as.numeric(churn1$InternetService)))/(max(as.numeric(churn1$InternetService)) - min(as.numeric(churn1$InternetService)))
churn1$OnlineSecurity <- (as.numeric(churn1$OnlineSecurity) - min(as.numeric(churn1$OnlineSecurity)))/(max(as.numeric(churn1$OnlineSecurity)) - min(as.numeric(churn1$OnlineSecurity)))
churn1$OnlineBackup <- (as.numeric(churn1$OnlineBackup) - min(as.numeric(churn1$OnlineBackup)))/(max(as.numeric(churn1$OnlineBackup)) - min(as.numeric(churn1$OnlineBackup)))
churn1$DeviceProtection <- (as.numeric(churn1$DeviceProtection) - min(as.numeric(churn1$DeviceProtection)))/(max(as.numeric(churn1$DeviceProtection)) - min(as.numeric(churn1$DeviceProtection)))
churn1$TechSupport <- (as.numeric(churn1$TechSupport) - min(as.numeric(churn1$TechSupport)))/(max(as.numeric(churn1$TechSupport)) - min(as.numeric(churn1$TechSupport)))
churn1$StreamingTV <- (as.numeric(churn1$StreamingTV) - min(as.numeric(churn1$StreamingTV)))/(max(as.numeric(churn1$StreamingTV)) - min(as.numeric(churn1$StreamingTV)))
churn1$StreamingMovies <- (as.numeric(churn1$StreamingMovies) - min(as.numeric(churn1$StreamingMovies)))/(max(as.numeric(churn1$StreamingMovies)) - min(as.numeric(churn1$StreamingMovies)))
churn1$Contract <- (as.numeric(churn1$Contract) - min(as.numeric(churn1$Contract)))/(max(as.numeric(churn1$Contract)) - min(as.numeric(churn1$Contract)))
churn1$PaperlessBilling <- (as.numeric(churn1$PaperlessBilling) - min(as.numeric(churn1$PaperlessBilling)))/(max(as.numeric(churn1$PaperlessBilling)) - min(as.numeric(churn1$PaperlessBilling)))
churn1$PaymentMethod <- (as.numeric(churn1$PaymentMethod) - min(as.numeric(churn1$PaymentMethod)))/(max(as.numeric(churn1$PaymentMethod)) - min(as.numeric(churn1$PaymentMethod)))
churn1$MonthlyCharges <- (as.numeric(churn1$MonthlyCharges) - min(as.numeric(churn1$MonthlyCharges)))/(max(as.numeric(churn1$MonthlyCharges)) - min(as.numeric(churn1$MonthlyCharges)))


#splitting data from Churn1

split_train_test1 <- createDataPartition(churn1$Churn,p=0.7,list=FALSE)
dtrain1 <- churn1[split_train_test1,]
dtest1 <-  churn1[-split_train_test1,]
dim(dtrain1)         
dim(dtest1)


#6.6.2 Model Development
set.seed(777)

nnm <- neuralnet(Churn~Contract+tenure+InternetService+MonthlyCharges
                 , data = na.omit(dtrain1), hidden = c(3,2), act.fct = "logistic", linear.output=FALSE)

plot(nnm, col.hidden = "darkgreen", 
     col.hidden.synapse = "darkgreen",
     show.weights = "F", 
     information = "F",
     fill = "pink")

#6.6.3 Prediction & Confusion Matrix

nnm_pred <- compute(nnm, dtest1[,-19])
p2 <- nnm_pred$net.result
nnm_pred2 <- ifelse(p2>0.8, 1, 0)
nnm_cm <- table(nnm_pred2, dtest1$Churn)


#6.6.4 Accuracy

nnm_ac <- sum(diag(nnm_cm)/nrow(dtest1))*100
nnm_ac


#7 ROC & AUC

# Which model is better?
# using ROC

# 1 logistic regression
roc_logit<-roc(dtest$Churn,predictor=factor(lr_pred1,ordered=TRUE), plot = TRUE)

# 2 Decision tree
roc_dtree <- roc(response = dtest$Churn, predictor = factor(tr_pred1, ordered = TRUE), plot = TRUE)

# 3 Random Forest
roc_random<-roc(response=dtest$Churn,predictor=factor(rf_pred2,ordered=TRUE), plot = TRUE)

# 4 Naive Bayes
roc_bayes<-roc(response=dtest$Churn,predictor=factor(nbm_pred,ordered=TRUE), plot = TRUE)

# 5 SVM
roc_svm<-roc(response=dtest$Churn,predictor=factor(svm_pred,ordered=TRUE), plot = TRUE)

# 6 ANN
roc_ann<-roc(response=dtest$Churn,predictor=factor(nnm_pred2,ordered=TRUE), plot = TRUE)



lines(roc_logit,col="red",lwd=4)
lines(roc_dtree,col="green",lwd=4)
lines(roc_random,col="cyan",lwd=4)
lines(roc_bayes,col="magenta",lwd=4)
lines(roc_svm,col="blue",lwd=4)
lines(roc_ann,col="black",lwd=4)

legend("right",
       legend=c("LR","DT","RF","NB","SVM", "ANN"),
       col=c("red", "green", "cyan","magenta","blue", "black"),
       lwd=4, cex =1.0, xpd = TRUE, horiz = FALSE)


# using AUC
LogisticRegression <- round((auc(roc_logit)*100),2)
DecisionTree <- round((auc(roc_dtree)*100),2)
RandomForest <- round((auc(roc_random)*100),2)
NaiveBayes <- round((auc(roc_bayes)*100),2)
SupportVectorMachine <- round((auc(roc_svm)*100),2)
NeuralNetwork <- round((auc(roc_ann)*100),2)

auc_df <- data.frame(LogisticRegression, DecisionTree, 
                     RandomForest, NaiveBayes, SupportVectorMachine, NeuralNetwork)
View(auc_df)


#8 Data Visualization based on Model results

# From Decision tree, we got Contract as the most important reason
# for customer churn

# From Logistic Regression, we got Tenure, Contract & PaperlessBilling
# as the most important reasons for customer churns

# Visualizing Contract

ggplot(churn, aes(x = Contract, fill = Churn)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) +
  labs(title="Churn rate by contract status")

# The churn rate of month-to-month contract customers is much higher
# than the longer contract customers. Customers who are more willing
# to commit to longer contracts are less likely to leave.

# Visualizing Tenure

ggplot(churn, aes(x = tenure, fill = Churn)) +
  geom_histogram(binwidth = 1) +
  labs(x = "Months",
       title = "Churn rate by tenure")

# the length of time as a customer decreases the likelihood of 
# churn. There is a large spike at 1 month, indicating that there 
# are a large portion of customers that will leave the after just 
# one month of service.

# Visualizing PaperlessBilling

ggplot(churn, aes(x = PaperlessBilling, fill = Churn)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) +
  labs(title="Churn rate by PaperlessBilling")

# The churn rate of customers who have paperlessbilling is higher.

# Visualizing MonthlyCharges


ggplot(churn, aes(x = MonthlyCharges, fill = Churn)) +
  geom_histogram(binwidth = 100) +
  labs(x = "Dollars",
       title = "Churn rate by MonthlyCharges")

#  customers who have spent more with the company tend not to 
# leave. This could just be a reflection of the tenure effect, 
# or it could be due to financial characteristics of the 
# customer: customers who are more financially well off are 
# less likely to leave.

# Visualizing InternetService

ggplot(churn, aes(x = InternetService, fill = Churn)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) +
  labs(title="Churn rate by internet service status")

# It appears as if customers with internet service are more 
# likely to churn than those that don't. This is more pronounced 
# for customers with fiber optic internet service, who are the 
# most likely to churn.


