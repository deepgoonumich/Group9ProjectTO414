---
title: "FinalProject"
author: "Pablo Suarez, Deep Goon, William Schwartz, Matthew Stuart, Maximilian Howarth"
date: "4/8/2020"
Outcome: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
#### Import all libraries
```{r}
library(class)
library(caret)
library(neuralnet)
library(kernlab)
```



## Loading and Exploring Data
```{r}
diabetes <- read.csv("diabetes.csv")
str(diabetes)
table(diabetes$Outcome)
```

## Data Cleaning, Randomization and Normalization
```{r}
# Randomize data
set.seed(42)
rows <- sample(nrow(diabetes))
diabetes <- diabetes[rows, ]

# Build Model Matrix to deal with all the factors (if any)
diabetes <- as.data.frame(model.matrix(~ . -1, data = diabetes))

# Remove NAs or impute missing values
diabetes$Pregnancies <- ifelse(is.na(diabetes$Pregnancies),mean(diabetes$Pregnancies,na.rm = TRUE),diabetes$Pregnancies)
diabetes$Glucose <- ifelse(is.na(diabetes$Glucose),mean(diabetes$Glucose,na.rm = TRUE),diabetes$Glucose)
diabetes$BloodPressure <- ifelse(is.na(diabetes$BloodPressure),mean(diabetes$BloodPressure,na.rm = TRUE),diabetes$BloodPressure)
diabetes$SkinThickness <- ifelse(is.na(diabetes$SkinThickness),mean(diabetes$SkinThickness,na.rm = TRUE),diabetes$SkinThickness)
diabetes$Insulin <- ifelse(is.na(diabetes$Insulin),mean(diabetes$Insulin,na.rm = TRUE),diabetes$Insulin)
diabetes$BMI <- ifelse(is.na(diabetes$BMI),mean(diabetes$BMI,na.rm = TRUE),diabetes$BMI)
diabetes$DiabetesPedigreeFunction <- ifelse(is.na(diabetes$DiabetesPedigreeFunction),mean(diabetes$DiabetesPedigreeFunction,na.rm = TRUE),diabetes$DiabetesPedigreeFunction)
diabetes$Age <- ifelse(is.na(diabetes$Age),mean(diabetes$Age,na.rm = TRUE),diabetes$Age)

# Normalize data
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

diabetes_norm <- as.data.frame(lapply(diabetes, normalize))

# Build train and test data
diabetes_train <- diabetes_norm[1:400, ]
diabetes_train_labels <- diabetes_norm[1:400, 9]
diabetes_test <- diabetes_norm[401:768, ]
diabetes_test_labels <- diabetes_norm[401:768, 9]
```

## Models

### Logistic Model
```
# Logistic Models

# First model
log1<- glm(formula = diabetes$Outcome ~., family="binomial", data=diabetes)
summary(log1)

# Second Model, AIC score reduced but marginally
log2<- glm(formula = diabetes$Outcome ~ Pregnancies + Glucose + BloodPressure + BMI + DiabetesPedigreeFunction  , family="binomial", data=diabetes)
summary(log2)
plot(log2)
```
### Linear Models || Current r square value: 0.2941
```{r}
# First Model
lm1 <- lm(diabetes$Outcome ~ ., data=diabetes)
summary(lm1)

# Second model (R squared value reduced)
lm2<-  lm(diabetes$Outcome ~ Pregnancies + Glucose + BloodPressure + BMI +  DiabetesPedigreeFunction, data = diabetes)
summary(lm2)
plot(lm2)

```

### KNN Model
```{r}
# Build first model
diabetes_test_pred <- knn(train = diabetes_train, test = diabetes_test,
                      cl = diabetes_train_labels, k=28)
confusionMatrix(table(diabetes_test_pred, diabetes_test_labels))

# Build second model with optimized K (use this result)
diabetes_test_pred <- knn(train = diabetes_train, test = diabetes_test,
                      cl = diabetes_train_labels, k=27)
confusionMatrix(table(diabetes_test_pred, diabetes_test_labels))

# Try Z score optimization
diabetes_z <- as.data.frame(scale(diabetes_norm))
diabetes_test_pred <- knn(train = diabetes_train, test = diabetes_test,
                      cl = diabetes_train_labels, k=202)
confusionMatrix(table(diabetes_test_pred, diabetes_test_labels))
diabetes_train <- diabetes_z[1:400, ]
diabetes_test <- diabetes_z[401:768, ]
diabetes_test_pred <- knn(train = diabetes_train, test = diabetes_test,
                      cl = diabetes_train_labels, k=27)
length(diabetes_test_pred)
confusionMatrix(table(diabetes_test_pred, diabetes_test_labels))
```
### ANN model  || Current highest accuracy achieved: 74.46%
```{r}
# Build train and test data
diabetes_train <- diabetes_norm[1:400, ]
diabetes_train_labels <- diabetes_norm[1:400, 9]
diabetes_test <- diabetes_norm[401:768, ]
diabetes_test_labels <- diabetes_norm[401:768, 9]

# simple ANN with only a single hidden neuron
diabetes_model <- neuralnet(formula = Outcome ~ ., data = diabetes_train)
plot(diabetes_model, rep = "best")

model_result =  compute(diabetes_model, diabetes_test)
predicted_strength = model_result$net.result

summary(predicted_strength)
binary_ps = ifelse(predicted_strength>-0.05247,1,0)
summary(diabetes_test)
# Build the confusion matrix


a  <- as.factor(diabetes_test$Outcome)
b <- as.factor(binary_ps)

confusionMatrix( b, a) ##Accuracy is 74.46% which is pretty low
```
### SVM Model || Current highest accuracy achieved: 73.64%
```{r}
# Build train and test data
diabetes_train <- diabetes_norm[1:400, ]
diabetes_train_labels <- diabetes_norm[1:400, 9]
diabetes_test <- diabetes_norm[401:768, ]
diabetes_test_labels <- diabetes_norm[401:768, 9]


diabetes_train$Outcome <- as.factor(diabetes_train$Outcome)
diabetes_test$Outcome <- as.factor(diabetes_test$Outcome)
diabetes_test_labels <- as.factor(diabetes_test_labels)

model_svm <- ksvm(diabetes_train$Outcome ~ ., data = diabetes_train, kernel = "rbfdot")

diabetes_predict <- predict(model_svm, diabetes_test)

confusionMatrix(diabetes_predict, diabetes_test_labels)  ##Accuracy is 73.64% which is pretty low
```