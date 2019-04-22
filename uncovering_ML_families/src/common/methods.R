#source("def_models.R")
library(R.utils)
library(caret)
#library(h2o)
options(warn=0)


#################### For run methods in Caret
runmethods <-function (mymethod, train, test) 
{
  
  ERROR <- FALSE
  model <- NA
  preds <- list()
 # print(paste("Dataset: ",datasets[selected],", Method: ", mymethod,", fold: ",ik))
  
  
  ########################
  ###  Decision Trees  ###
  ######################## 
  
  
  
  if (mymethod=="c5.0") {
    load_install("C50")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- C5.0(Class ~., data = train, trace=TRUE), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:C50", unload=TRUE), error = function(e){print(e)})
  }
  
  # winnow = TRUE -> pre-selection of attributes that will be used to contruct the decision tree/ruleset
  if (mymethod=="c5.0_winnow"){
    load_install("C50")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- C5.0(Class ~., data = train, control = C5.0Control(winnow = TRUE)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:C50", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod=="J48"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- J48(Class ~., data = train, control = Weka_control(U = FALSE,A=TRUE)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})        
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod=="J48Unp"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- J48(Class ~., data = train, control = Weka_control(U = TRUE,A=TRUE)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  # Classifier for building 'logistic model trees', which are classification trees with logistic regression functions at the leaves. The algorithm can deal with binary and multi-class target variables, numeric and nominal attributes and missing values.
  if (mymethod=="LMT"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- LMT(Class ~., data = train), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})       
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  # Use cross-validation for boosting at all nodes (i.e., disable heuristic)
  if (mymethod=="LMT_CV"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- LMT(Class ~., data = train, control = Weka_control(C = TRUE)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  # The AIC is used to choose the best iteration.
  if (mymethod=="LMT_AIC"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- LMT(Class ~., data = train, control = Weka_control(A = TRUE)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  #Recursive Partitioning and Regression Trees
  if (mymethod=="rpart") {
    load_install("rpart")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- rpart(Class ~., data = train, method = "class"), error = function(e) {print(e);return(TRUE)})
    #ERROR <-  tryCatch(model <- rpart(Class ~., data = train, preProc = c("center", "scale"), method = "class"), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
      if (is.matrix(preds)){
        print("IS MATRIX")
        #preds <-  as.factor(as.vector(apply(preds,1,function (x) {colnames(preds)[which.max(x)]})))
        preds <-  factor(as.vector(apply(preds,1,function (x) {colnames(preds)[which.max(x)]})), levels=levels(as.factor(train$Class)))
        #Diu que els levels differeixen, però no és cert. No deuria donar problemes amb el dendrograma, ja que estan tots els levels al incloure l'entrenament
        # Si dona problemes a traure el dendrograma, revisar aquesta part. Si no, llevem el model. Es un CART i hi ha arbres de sobra
        #levels(preds) <- levels(train)
      }
    }
    tryCatch(detach("package:rpart", unload=TRUE), error = function(e){print(e)})
    # printcp(Fit2) # display the results
    # plotcp(Fit2) # visualize cross-validation results
    # summary(Fit2) # detailed summary of splits
    # testPred <- predict(Fit2, test)
    # testPred <- as.factor(as.vector(apply(testPred,1,function (x) {colnames(testPred)[which.max(x)]})))
  }
  
  # conditional inference trees by recursively making binary splittings on the variables with the highest association to the class (measured by a statistical test). The threshold in the association measure is given by the parameter mincriterion
  if (mymethod == "ctree_c0.01"){
    load_install("party")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ctree", tuneGrid=data.frame(mincriterion=0.01), trControl = trainControl(method="none")), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {print(e); return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:party", unload=TRUE), error = function(e){print(e)})
  }
  
  # mincriterion = 0.5
  if (mymethod == "ctree_c0.05"){
    load_install("party")
    print("Training")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ctree", tuneGrid=data.frame(mincriterion=0.5), trControl = trainControl(method="none")), error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {print("ExcepTION: "); print(e); return(rep(FALSE, nrow(test)))})

    }
    tryCatch(detach("package:party", unload=TRUE), error = function(e){print(e)})
  }
  
  # mincriterion = 0.99
  if (mymethod == "ctree_c0.99"){
    load_install("party")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ctree", tuneGrid=data.frame(mincriterion=0.99), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:party", unload=TRUE), error = function(e){print(e)})
  }
  
  ##########################
  ### Rule-based methods ###
  ##########################
  
  # RWeka algorithm; JRip implements a propositional rule learner
  if (mymethod=="JRip"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <-JRip(Class ~., data = train), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  # RWeka algorithm; JRip implements a propositional rule learner (unpruned)
  if (mymethod=="JRip_Unp"){
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    load_install("RWeka")
    ERROR <-  tryCatch(model <-JRip(Class ~., data = train, control = Weka_control(E = TRUE)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  # RWeka algorithm: PART generates PART decision lists using the approach of Frank and Witten 
  if (mymethod == "PART"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- PART(Class ~., data = train), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  #############################
  ### Discriminant analysis ###
  #############################
  
  if (mymethod == "rlda") {
    load_install("sparsediscrim")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    #ERROR <-  tryCatch(model <- rrlda(train[,-ncol(train)], train[,ncol(train)]), error = function(e) {print(e);return(TRUE)})
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rlda"), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
      
    }else{
      preds<- tryCatch(predict(model, test[,1:(ncol(test)-1)]), error = function(e) {return(rep(FALSE, nrow(test)))})
      if (is.list(preds)){
        preds <-  preds$class
      }
    }
    tryCatch(detach("package:sparsediscrim", unload=TRUE), error = function(e){print(e)})
  }
  
  
  # It performs LDA or diagonal discriminant analysis (DDA) with variable selection using CAT (Correlation-Adjusted T) scores. The best classifier (LDA or DDA) is selected.
  if (mymethod == "sda_L0.0") {
    load_install("sda")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "sda", tuneGrid=data.frame(diagonal = FALSE, lambda = 0.0), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    print(ERROR)
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:sda", unload=TRUE), error = function(e){print(e)})
  }
  # lambda = 0.5
  if (mymethod == "sda_L0.5") {
    load_install("sda")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "sda", tuneGrid=data.frame(diagonal = FALSE, lambda = 0.5), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:sda", unload=TRUE), error = function(e){print(e)})
  }
  
  # lambda = 1
  if (mymethod == "sda_L1.0") {
    load_install("sda")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "sda", tuneGrid=data.frame(diagonal = FALSE, lambda = 1.0), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:sda", unload=TRUE), error = function(e){print(e)})
  }
  
  # quadratic discriminant analysis
  # if (mymethod == "qda") {
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "qda", trControl = trainControl(method="none"))
  #   preds<-predict(model, newdata = test)
  # }
  # 
  # # Robust QDA
  # if (mymethod == "QdaCov") {
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "qda", trControl = trainControl(method="none"))
  #   preds<-predict(model, newdata = test)
  # }
  
  # flexible discriminant analysis. prune = 2
  if (mymethod == "fda_prune2") {
    load_install("mda")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "fda", tuneGrid=data.frame(nprune = 2, degree = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:mda", unload=TRUE), error = function(e){print(e)})
  }
  
  # flexible discriminant analysis. prune = 9
  if (mymethod == "fda_prune9") {
    load_install("mda")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "fda", tuneGrid=data.frame(nprune = 9, degree = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:mda", unload=TRUE), error = function(e){print(e)})
  }
  
  # flexible discriminant analysis. prune = 17
  if (mymethod == "fda_prune17") {
    load_install("mda")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "fda", tuneGrid=data.frame(nprune = 17, degree = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:mda", unload=TRUE), error = function(e){print(e)})
  }
  
  # mixture discriminant analysis. subclasses = 2
  if (mymethod == "mda_subc2") {
    load_install("mda")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mda", tuneGrid=data.frame(subclasses = 2), trControl = trainControl(method="none")), error = function(e) {print(e);return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:mda", unload=TRUE), error = function(e){print(e)})
  }
  
  # subclasses = 3
  if (mymethod == "mda_subc3") {
    load_install("mda")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mda", tuneGrid=data.frame(subclasses = 3), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:mda", unload=TRUE), error = function(e){print(e)})
  }
  
  # subclasses = 4
  if (mymethod == "mda_subc4") {
    load_install("mda")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mda", tuneGrid=data.frame(subclasses = 4), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:mda", unload=TRUE), error = function(e){print(e)})
  }
  
  
  ################
  ### Bayesian ###
  ################
  
  if (mymethod == "W_NB")
  {
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    NB <- make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
    ERROR <-  tryCatch(model <- NB(Class ~., data = train), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "NB") {
    load_install("naivebayes")
    preProc <- preProcess(train, method=c("center", "scale"))
    test <- predict(preProc, test)
    train <- predict(preProc, train)
    ERROR <-  tryCatch(model <- naive_bayes(train[,-ncol(train)], train[,ncol(train)]), error = function(e) {print(e);return(TRUE)})
    #ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, method = "naive_bayes"), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      #preds<- tryCatch(predict(model, test[,1:ncol(test)-1]), error = function(e) {return(rep(FALSE, nrow(test)))})
      preds<- tryCatch(predict(model, test), error = function(e) {return(rep(FALSE, nrow(test)))})
      print(preds)
    }
    tryCatch(detach("package:naivebayes", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "NB_laplace") {
    load_install("naivebayes")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- naive_bayes(train[,-ncol(train)], train[,ncol(train)], laplace = 3), error = function(e) {print(e);return(TRUE)})
    #ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, method = "naive_bayes", tuneGrid=data.frame(laplace=3, usekernel=NA, adjust=NA)), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:naivebayes", unload=TRUE), error = function(e){print(e)})
  }
  
  
  
  #######################
  ### Neural Networks ###
  #######################
  
  # Radial Basis Function Network with negative threshold
  if (mymethod == "rbf") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rbfDDA", tuneGrid=data.frame(negativeThreshold=0.001), trControl = trainControl(method="none")), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload=TRUE), error = function(e){print(e)})
  }
  
  # Multilayer perceptron wiht 1 unit in the hidden layer
  if (mymethod == "mlp_1") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlp", tuneGrid=data.frame(size=1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload=TRUE), error = function(e){print(e)})
  }
  
  # Multilayer perceptron wiht 3 unit in the hidden layer
  if (mymethod == "mlp_3") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlp", tuneGrid=data.frame(size=3), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload=TRUE), error = function(e){print(e)})
  }
  
  # Multilayer perceptron wiht 5 unit in the hidden layer
  if (mymethod == "mlp_5") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlp", tuneGrid=data.frame(size=5), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload=TRUE), error = function(e){print(e)})
  }
  
  # Multilayer perceptron wiht 7 unit in the hidden layer
  if (mymethod == "mlp_7") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlp", tuneGrid=data.frame(size=7), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload=TRUE), error = function(e){print(e)})
  }
  
  # Multilayer perceptron wiht 9 unit in the hidden layer
  if (mymethod == "mlp_9") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlp", tuneGrid=data.frame(size=9), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload=TRUE), error = function(e){print(e)})
  }
  
  # committee of 5 MLPs (the number of MLPs is given by parameter repeat) trained with different random weight initializations and bag=false. decay = 1e-04
  if (mymethod == "avNNet_decay1e04") {
    load_install("nnet")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "avNNet", tuneGrid=data.frame(size=5, decay = 1e-04, bag = FALSE ), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:nnet", unload=TRUE), error = function(e){print(e)})
  }
  
  # committee of 5 MLPs (the number of MLPs is given by parameter repeat) trained with different random weight initializations and bag=false. decay = 1e-03
  if (mymethod == "avNNet_decay1e03") {
    load_install("nnet")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "avNNet", tuneGrid=data.frame(size=5, decay = 1e-03, bag = FALSE ), trControl = trainControl(method="none")), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:nnet", unload=TRUE), error = function(e){print(e)})
  }
  
  # committee of 5 MLPs (the number of MLPs is given by parameter repeat) trained with different random weight initializations and bag=false. decay = 1e-02
  if (mymethod == "avNNet_decay1e02") {
    load_install("nnet")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "avNNet", tuneGrid=data.frame(size=5, decay = 1e-02, bag = FALSE ), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:nnet", unload=TRUE), error = function(e){print(e)})
  }
  
  # committee of 5 MLPs (the number of MLPs is given by parameter repeat) trained with different random weight initializations and bag=false. decay = 0.1
  if (mymethod == "avNNet_decay01") {
    load_install("nnet")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "avNNet", tuneGrid=data.frame(size=5, decay = 0.1, bag = FALSE ), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:nnet", unload=TRUE), error = function(e){print(e)})
  }
  
  # committee of 5 MLPs (the number of MLPs is given by parameter repeat) trained with different random weight initializations and bag=false. decay = 0
  if (mymethod == "avNNet_decay0") {
    load_install("nnet")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "avNNet", tuneGrid=data.frame(size=5, decay = 0, bag = FALSE ), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:nnet", unload=TRUE), error = function(e){print(e)})
  }
  
  # committee of 5 MLPs (the number of MLPs is given by parameter repeat) trained with different random weight initializations and bag=false. decay = 1
  if (mymethod == "avNNet_decay1") {
    load_install("nnet")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "avNNet", tuneGrid=data.frame(size=5, decay = 1, bag = FALSE ), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:nnet", unload=TRUE), error = function(e){print(e)})
  }
  
  # trains the MLP using caret and the nnet package, but running principal component analysis (PCA) previously on the data set
  if (mymethod == "pcaNNet") {
    load_install("nnet")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "pcaNNet", tuneGrid=data.frame(size=5, decay = 1e-04), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:nnet", unload=TRUE), error = function(e){print(e)})
  }
  
  # learning vector quantization implemented using the function lvq in the class package, with codebook of size 50, and k=1 nearest neighbors.
  if (mymethod == "lvq_1") {
    load_install("class")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "lvq", tuneGrid=data.frame(size=50, k = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:class", unload=TRUE), error = function(e){print(e)})
  }
  
  # k=3 nearest neighbors.
  if (mymethod == "lvq_3") {
    load_install("class")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "lvq", tuneGrid=data.frame(size=50, k = 3), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:class", unload=TRUE), error = function(e){print(e)})
  }
  
  # k=5 nearest neighbors.
  if (mymethod == "lvq_5") {
    load_install("class")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "lvq", tuneGrid=data.frame(size=50, k = 5), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:class", unload=TRUE), error = function(e){print(e)})
  }
  
  ###############################
  ### Support Vector Machines ###
  ###############################
  
  # SMO implements John C. Platt's sequential minimal optimization algorithm for training a support vector classifier using polynomial or RBF kernels. Multi-class problems are solved using pairwise classification.
  if (mymethod == "SMO") {
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- SMO(Class ~., data = train), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "svmRadialCost") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost"), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "svmRadialCost_C2_-5") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^(-5)), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }

  if (mymethod == "svmRadialCost_C2_-3") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^(-3)), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "svmRadialCost_C2_-1") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^(-1)), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  
  # cost C = 0.1
  if (mymethod == "svmRadialCost_C0.1") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProcess = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=0.1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # cost C = 1
  if (mymethod == "svmRadialCost_C1") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProcess = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    #ERROR <-  tryCatch(model <- ksvm(Class ~ ., data = train, scaled=TRUE, C=100, kernel="rbfdot"), error = function(e) {return(TRUE)})  
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  } 
  
  if (mymethod == "svmRadialCost_C2_1") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2), trControl = trainControl(method="none")), error = function(e) {print(e);return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "svmRadialCost_C2_3") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^3), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "svmRadialCost_C2_5") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^5), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "svmRadialCost_C2_7") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^7), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "svmRadialCost_C2_9") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^9), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "svmRadialCost_C2_11") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^11), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "svmRadialCost_C2_13") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^13), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "svmRadialCost_C2_15") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^15), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # cost C = 2
  if (mymethod == "svmRadialCost_C2") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProcess = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # cost C = 10
  if (mymethod == "svmRadialCost_C10") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProcess = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=10), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # cost C = 100
  if (mymethod == "svmRadialCost_C100") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProcess = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=2^12), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # cost C = 1000
  if (mymethod == "svmRadialCost_C1000") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProcess = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=1000), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }


  if (mymethod == "svmRadialCost_C10000") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProcess = c("center", "scale"), method = "svmRadialCost", tuneGrid=data.frame(C=10000), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # uses the function ksvm (kernlab package) with linear kernel tuning C =0.01
  if (mymethod == "svmLinear_C0.01") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmLinear", tuneGrid=data.frame(C=0.01), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  # cost C = 0.1
  if (mymethod == "svmLinear_C0.1") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmLinear", tuneGrid=data.frame(C=0.1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # cost C = 1
  if (mymethod == "svmLinear_C1") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmLinear", tuneGrid=data.frame(C=1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # cost C = 2
  if (mymethod == "svmLinear_C2") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmLinear", tuneGrid=data.frame(C=2), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # cost C = 4
  if (mymethod == "svmLinear_C4") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmLinear", tuneGrid=data.frame(C=4), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # cost C = 8
  if (mymethod == "svmLinear_C8") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmLinear", tuneGrid=data.frame(C=8), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # linear, quadratic and cubic kernels. Degree = 1. Scale = 0.001
  if (mymethod == "svmPoly_d_1_s_0.001") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmPoly", tuneGrid=data.frame(degree=1, scale= 0.001, C = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # linear, quadratic and cubic kernels. Degree = 1. Scale = 0.01
  if (mymethod == "svmPoly_d_1_s_0.01") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmPoly", tuneGrid=data.frame(degree=1, scale= 0.01, C = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  # linear, quadratic and cubic kernels. Degree = 1. Scale = 0.1
  if (mymethod == "svmPoly_d_1_s_0.1") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmPoly", tuneGrid=data.frame(degree=1, scale= 0.1, C = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # linear, quadratic and cubic kernels. Degree = 2. Scale = 0.001
  if (mymethod == "svmPoly_d_2_s_0.001") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmPoly", tuneGrid=data.frame(degree=2, scale= 0.001, C = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # linear, quadratic and cubic kernels. Degree = 2. Scale = 0.01
  if (mymethod == "svmPoly_d_2_s_0.01") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmPoly", tuneGrid=data.frame(degree=2, scale= 0.01, C = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # linear, quadratic and cubic kernels. Degree = 2. Scale = 0.1
  if (mymethod == "svmPoly_d_2_s_0.1") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmPoly", tuneGrid=data.frame(degree=2, scale= 0.1, C = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # linear, quadratic and cubic kernels. Degree = 3. Scale = 0.001
  if (mymethod == "svmPoly_d_3_s_0.001") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmPoly", tuneGrid=data.frame(degree=3, scale= 0.001, C = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # linear, quadratic and cubic kernels. Degree = 3. Scale = 0.01
  if (mymethod == "svmPoly_d_3_s_0.01") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmPoly", tuneGrid=data.frame(degree=3, scale= 0.01, C = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  # linear, quadratic and cubic kernels. Degree = 3. Scale = 0.1 #62
  if (mymethod == "svmPoly_d_3_s_0.1") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmPoly", tuneGrid=data.frame(degree=3, scale= 0.1, C = 1), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload=TRUE), error = function(e){print(e)})
  }
  
  ##################
  ###  Boosting  ###
  ##################
  
  # adaboost.M1 method (Freund and Schapire, 1996) to create an adaboost ensemble of classification trees 
  # DOES NOT SUPPORT MULTICLASS PROBLEMS
  
  # if (mymethod == "ada_iter_50_depth_1") {
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ada", tuneGrid=data.frame(iter = 50,maxdepth= 1, nu=0.1), trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # if (mymethod == "ada_iter_50_depth_2") {
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ada", tuneGrid=data.frame(iter = 50,maxdepth= 2, nu=0.1), trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # if (mymethod == "ada_iter_50_depth_3") {
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ada", tuneGrid=data.frame(iter = 50,maxdepth= 3, nu=0.1), trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # 
  # if (mymethod == "ada_iter_100_depth_1") {
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ada", tuneGrid=data.frame(iter = 100,maxdepth= 1, nu=0.1), trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # if (mymethod == "ada_iter_100_depth_2") {
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ada", tuneGrid=data.frame(iter = 100,maxdepth= 2, nu=0.1), trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # if (mymethod == "ada_iter_100_depth_3") {
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ada", tuneGrid=data.frame(iter = 100,maxdepth= 3, nu=0.1), trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # 
  # if (mymethod == "ada_iter_150_depth_1") {
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ada", tuneGrid=data.frame(iter = 150,maxdepth= 1, nu=0.1), trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # if (mymethod == "ada_iter_150_depth_2") {
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ada", tuneGrid=data.frame(iter = 150,maxdepth= 2, nu=0.1), trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # if (mymethod == "ada_iter_150_depth_3") { #71
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ada", tuneGrid=data.frame(iter = 150,maxdepth= 3, nu=0.1), trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  
  # uses additive logistic regressors (DecisionStump) base learners, the 100% of weight mass to base training on, without cross-validation, one run for internal cross-validation, threshold 1.79 on likelihood improvement, shrinkage parameter 1, and 10 iterations.
  # if (mymethod == "logitBoost_i11") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "LogitBoost", tuneGrid=data.frame(nIter = 11), trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # 
  # if (mymethod == "logitBoost_i21") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "LogitBoost", tuneGrid=data.frame(nIter = 21), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
  #    preds <- predict(model, test)
  #  }else{
  #    preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
  #  }
   
  # if (mymethod == "logitBoost_i31") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "LogitBoost", tuneGrid=data.frame(nIter = 31), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
  #    preds <- predict(model, test)
  # }
  # else{
  #    preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
  # }
  
  if (mymethod == "gbm_1_50") { 
    load_install("gbm")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gbm", tuneGrid=data.frame(n.trees = 50, interaction.depth = 1, shrinkage = 0.1 , n.minobsinnode = 10), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gbm", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "gbm_1_100") {
    load_install("gbm")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gbm", tuneGrid=data.frame(n.trees = 100, interaction.depth = 1, shrinkage = 0.1 , n.minobsinnode = 10), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gbm", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "gbm_1_150") {
    load_install("gbm")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gbm", tuneGrid=data.frame(n.trees = 150, interaction.depth = 1, shrinkage = 0.1 , n.minobsinnode = 10), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gbm", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "gbm_2_50") {
    load_install("gbm")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gbm", tuneGrid=data.frame(n.trees = 50, interaction.depth = 2, shrinkage = 0.1 , n.minobsinnode = 10), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gbm", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "gbm_2_100") {
    load_install("gbm")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gbm", tuneGrid=data.frame(n.trees = 100, interaction.depth = 2, shrinkage = 0.1 , n.minobsinnode = 10), trControl = trainControl(method="none")), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gbm", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "gbm_2_150") {
    load_install("gbm")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gbm", tuneGrid=data.frame(n.trees = 150, interaction.depth = 2, shrinkage = 0.1 , n.minobsinnode = 10), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gbm", unload=TRUE), error = function(e){print(e)})
  }
  
  
  if (mymethod == "gbm_3_50") {
    load_install("gbm")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gbm", tuneGrid=data.frame(n.trees = 50, interaction.depth = 3, shrinkage = 0.1 , n.minobsinnode = 10), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gbm", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "gbm_3_100") {
    load_install("gbm")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gbm", tuneGrid=data.frame(n.trees = 100, interaction.depth = 3, shrinkage = 0.1 , n.minobsinnode = 10), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gbm", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "gbm_3_150") {
    load_install("gbm")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gbm", tuneGrid=data.frame(n.trees = 150, interaction.depth = 3, shrinkage = 0.1 , n.minobsinnode = 10), trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gbm", unload=TRUE), error = function(e){print(e)})
  }
  
  # # Boosted Tree: Gradient boosting for optimizing arbitrary loss functions where regression trees are utilized as base-learners.  maxdepth = 1
  # if (mymethod == "blackboost_maxdepth1") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "blackboost",tuneGrid=data.frame(mstop = 50, maxdepth= 1),  trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # 
  # # Boosted Tree maxdepth = 2
  # if (mymethod == "blackboost_maxdepth2") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "blackboost",tuneGrid=data.frame(mstop = 50, maxdepth= 2),  trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # # Boosted Tree maxdepth = 3
  # if (mymethod == "blackboost_maxdepth3") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "blackboost",tuneGrid=data.frame(mstop = 50, maxdepth= 3),  trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  
  
  
  #################
  ###  Bagging  ###
  #################
  if (mymethod == "bag") { 
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bag", trControl = trainControl(method="none"), trace=TRUE), error = function(e) {print(e); return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }
  
  if (mymethod == "bagEarth") {
    load_install("earth")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bagEarth", trControl = trainControl(method="none"), trace=TRUE), error = function(e) {print(e);return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:earth", unload=TRUE), error = function(e){print(e)})
  }

  if (mymethod == "treebag") {
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "treebag", trControl = trainControl(method="none")), error = function(e) {print(e); return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }
  # Bagged Flexible Discriminant Analysis nprune =2
  if (mymethod == "bagFDA_prune2") {
    load_install("earth")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bagFDA",tuneGrid=data.frame(degree = 2, nprune= 2),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    #ERROR <-  tryCatch(model <- caret::train(train[,-ncol(train)], train[,ncol(train)], data = train, preProc = c("center", "scale"), method = "bagFDA",tuneGrid=data.frame(degree = 2, nprune= 2),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:earth", unload=TRUE), error = function(e){print(e)})
  }
  # nprune = 4
  if (mymethod == "bagFDA_prune4") { 
    load_install("earth")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bagFDA",tuneGrid=data.frame(degree = 2, nprune= 4),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:earth", unload=TRUE), error = function(e){print(e)})
  }
  
  # nprune = 8
  if (mymethod == "bagFDA_prune8") { 
    load_install("earth")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bagFDA",tuneGrid=data.frame(degree = 2, nprune= 8),  trControl = trainControl(method="none")), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:earth", unload=TRUE), error = function(e){print(e)})
  }
  # nprune = 16  #82
  if (mymethod == "bagFDA_prune16") { 
    load_install("earth")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bagFDA",tuneGrid=data.frame(degree = 2, nprune= 16),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:earth", unload=TRUE), error = function(e){print(e)})
  }
  
  ##################
  ###  Stacking  ###
  ##################
  
  ########################
  ###  Random Forests  ###
  ########################
  
  # creates a random forest with mtry = 2
  if (mymethod == "rf") {
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rf",  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # creates a random forest with mtry = 2
  if (mymethod == "rf_mtry2") {
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rf",tuneGrid=data.frame(mtry = 2),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  # creates a random forest with mtry = 4
  if (mymethod == "rf_mtry4") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rf",tuneGrid=data.frame(mtry = 4),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  # creates a random forest with mtry = 8
  if (mymethod == "rf_mtry8") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rf",tuneGrid=data.frame(mtry = 8),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }
  # creates a random forest with mtry = 16
  if (mymethod == "rf_mtry16") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rf",tuneGrid=data.frame(mtry = 16),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  # creates a random forest with mtry = 32
  if (mymethod == "rf_mtry32") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rf",tuneGrid=data.frame(mtry = 32),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  # creates a random forest with mtry = 64
  if (mymethod == "rf_mtry64") {
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rf",tuneGrid=data.frame(mtry = 64),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  # creates a random forest with mtry = 128  
  if (mymethod == "rf_mtry128") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rf",tuneGrid=data.frame(mtry = 128),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # creates a Regularised random forest with mtry = 2
  if (mymethod == "rrf_mtry2") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "RRF",tuneGrid=data.frame(mtry = 2, coefReg=1, coefImp = 1),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # creates a Regularised random forest with mtry = 4
  if (mymethod == "rrf_mtry4") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "RRF",tuneGrid=data.frame(mtry = 4, coefReg=1, coefImp = 1),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  # creates a Regularised random forest with mtry = 8
  if (mymethod == "rrf_mtry8") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "RRF",tuneGrid=data.frame(mtry = 8, coefReg=1, coefImp = 1),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # creates a Regularised random forest with mtry = 16
  if (mymethod == "rrf_mtry16") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "RRF",tuneGrid=data.frame(mtry = 16, coefReg=1, coefImp = 1),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # creates a Regularised random forest with mtry = 32
  if (mymethod == "rrf_mtry32") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "RRF",tuneGrid=data.frame(mtry = 32, coefReg=1, coefImp = 1),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # creates a Regularised random forest with mtry = 64
  if (mymethod == "rrf_mtry64") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "RRF",tuneGrid=data.frame(mtry = 64, coefReg=1, coefImp = 1),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # creates a Regularised random forest with mtry = 128
  if (mymethod == "rrf_mtry128") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "RRF",tuneGrid=data.frame(mtry = 128, coefReg=1, coefImp = 1),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  
  # is a random forest and bagging ensemble of conditional inference trees (ctrees) aggregated by averaging observation weights extracted from each ctree. with mtry = 2
  if (mymethod == "cforest_mtry2") {
    load_install("party")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "cforest",tuneGrid=data.frame(mtry = 2),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:party", unload=TRUE), error = function(e){print(e)})
  }
  
  # mtry = 4
  if (mymethod == "cforest_mtry4") { 
    load_install("party")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "cforest",tuneGrid=data.frame(mtry = 4),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:party", unload=TRUE), error = function(e){print(e)})
  }
  
  # mtry = 8
  if (mymethod == "cforest_mtry8") {
    load_install("party")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "cforest",tuneGrid=data.frame(mtry = 8),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:party", unload=TRUE), error = function(e){print(e)})
  }
  
  # mtry = 16
  if (mymethod == "cforest_mtry16") { 
    load_install("party")
    print(paste("Train data:", nrow(train), "instances,", ncol(train), "feats"))
    
    print("Training...")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "cforest",tuneGrid=data.frame(mtry = 16),  trControl = trainControl(method="none")), error = function(e) {print(e);return(TRUE)})
    print("Trained")
    print(ERROR)
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      
      print("Predicting...")
      print(paste("Test data:", nrow(test), "instances,", ncol(test), "feats"))
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
      print("Predicted...")
      print(paste("Number of labels:", length(preds)))
      #print(preds)
    }
    tryCatch(detach("package:party", unload=TRUE), error = function(e){print(e)})
  }
  # mtry = 32
  if (mymethod == "cforest_mtry32") { 
    load_install("party")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "cforest",tuneGrid=data.frame(mtry = 32),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:party", unload=TRUE), error = function(e){print(e)})
  }
  
  # mtry = 64
  if (mymethod == "cforest_mtry64") { 
    load_install("party")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "cforest",tuneGrid=data.frame(mtry = 64),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:party", unload=TRUE), error = function(e){print(e)})
  }
  
  # mtry = 128
  if (mymethod == "cforest_mtry128") { 
    load_install("party")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "cforest",tuneGrid=data.frame(mtry = 128),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:party", unload=TRUE), error = function(e){print(e)})
  }
  
  # is a random forest and bagging ensemble of conditional inference trees (ctrees) aggregated by averaging observation weights extracted from each ctree. with mtry = 2
  if (mymethod == "parRF_mtry2") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "parRF",tuneGrid=data.frame(mtry = 2),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # mtry = 4
  if (mymethod == "parRF_mtry4") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "parRF",tuneGrid=data.frame(mtry = 4),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # mtry = 8
  if (mymethod == "parRF_mtry8") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "parRF",tuneGrid=data.frame(mtry = 8),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # mtry = 16
  if (mymethod == "parRF_mtry16") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "parRF",tuneGrid=data.frame(mtry = 16),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  # mtry = 32
  if (mymethod == "parRF_mtry32") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "parRF",tuneGrid=data.frame(mtry = 32),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  # mtry = 64
  if (mymethod == "parRF_mtry64") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "parRF",tuneGrid=data.frame(mtry = 64),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  # mtry = 128  #110
  if (mymethod == "parRF_mtry128") { 
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "parRF",tuneGrid=data.frame(mtry = 128),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload=TRUE), error = function(e){print(e)})
  }
  
  ################################
  ### Nearest neighbor methods ###
  ################################
  
  ee <- function(){ return(TRUE)}
  
  if (mymethod == "knn_k1") { 
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "knn",tuneGrid=data.frame(k = 1),  trControl = trainControl(method="none")), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }
  
  if (mymethod == "knn_k2") { 
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "knn",tuneGrid=data.frame(k = 2),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }
  
  if (mymethod == "knn_k3") { 
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "knn",tuneGrid=data.frame(k = 3),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }
  
  
  if (mymethod == "knn_k5") { 
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "knn",tuneGrid=data.frame(k = 5),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }
  
  
  if (mymethod == "knn_k7") { 
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "knn",tuneGrid=data.frame(k = 7),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }
  
  
  if (mymethod == "knn_k9") { 
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "knn",tuneGrid=data.frame(k = 9),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }
  
  if (mymethod == "Ibk_k1") { 
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- IBk(Class ~., data = train,control = Weka_control(K=1)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "Ibk_k2") { 
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- IBk(Class ~., data = train,control = Weka_control(K=2)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  
  if (mymethod == "Ibk_k3") { 
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- IBk(Class ~., data = train,control = Weka_control(K=3)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "Ibk_k5") { 
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- IBk(Class ~., data = train,control = Weka_control(K=5)), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "Ibk_k7") {
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- IBk(Class ~., data = train,control = Weka_control(K=7)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "Ibk_k9") { #122
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- IBk(Class ~., data = train,control = Weka_control(K=9)), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  
  ################################################################
  ### Partial least squares and principal component regression ###
  ################################################################
  
  # uses the function mvr in the pls package to fit a PLSR (Martens, 1989) model tuning the number of components from 1 to 10.
  if (mymethod == "pls_ncomp1") { 
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "pls",tuneGrid=data.frame(ncomp = 1),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:pls", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "pls_ncomp2") { 
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "pls",tuneGrid=data.frame(ncomp = 2),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:pls", unload=TRUE), error = function(e){print(e)})
  }
  
  # if (mymethod == "pls_ncomp3") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "pls",tuneGrid=data.frame(ncomp = 3),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
  #   
  #    if (is.logical(ERROR)){
  #     preds <- rep(FALSE, nrow(test))
  #   }else{
  #     preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
  #   }
  # }
  # 
  # if (mymethod == "pls_ncomp5") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "pls",tuneGrid=data.frame(ncomp = 5),  trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # 
  # if (mymethod == "pls_ncomp10") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "pls",tuneGrid=data.frame(ncomp = 10),  trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  # 
  if (mymethod == "simpls") { 
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "simpls",trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    } 
    tryCatch(detach("package:pls", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "simpls_ncomp1") { 
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "simpls",tuneGrid=data.frame(ncomp = 1),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    } 
    tryCatch(detach("package:pls", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "simpls_ncomp2") { 
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "simpls",tuneGrid=data.frame(ncomp = 2),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    } 
    tryCatch(detach("package:pls", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "simpls_ncomp3") { 
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "simpls",tuneGrid=data.frame(ncomp = 3),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    } 
    tryCatch(detach("package:pls", unload=TRUE), error = function(e){print(e)})
  }
  
  
  if (mymethod == "simpls_ncomp4") { 
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "simpls",tuneGrid=data.frame(ncomp = 4),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    } 
    tryCatch(detach("package:pls", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "simpls_ncomp7") { 
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "simpls",tuneGrid=data.frame(ncomp = 7),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    } 
    tryCatch(detach("package:pls", unload=TRUE), error = function(e){print(e)})
  }
  
  # if (mymethod == "simpls_ncomp3") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "simpls",tuneGrid=data.frame(ncomp = 3),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
  #     testF <<- test
  #     trainF <<- train
  #     modelF <<- modelF
  #     print(ERROR)
  #    if (is.logical(ERROR)){
  #     preds <- rep(FALSE, nrow(test))
  #   }else{
  #     preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
  #     print(preds)
  #   }
  # }
  
  # if (mymethod == "simpls_ncomp5") { 
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "simpls",tuneGrid=data.frame(ncomp = 5),  trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  #   
  # if (mymethod == "simpls_ncomp10") { #132
  #    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "simpls",tuneGrid=data.frame(ncomp = 10),  trControl = trainControl(method="none"))
  #   preds <- predict(model, test)
  # }
  
  ###########################################
  ### Logistic and multinomial regression ###
  ###########################################
  
  
  # RWeka algorithm; Logistic Regression: César
  if (mymethod=="LR"){ 
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <-Logistic(Class ~., data = train), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "gcvEarth") { 
    load_install("earth")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gcvEarth",  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:earth", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "gcvEarth_d1") { 
    load_install("earth")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gcvEarth",tuneGrid=data.frame(degree = 1),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:earth", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "gcvEarth_d2") { 
    load_install("earth")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gcvEarth",tuneGrid=data.frame(degree = 2),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:earth", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "gcvEarth_d3") { #135
    load_install("earth")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gcvEarth",tuneGrid=data.frame(degree = 3),  trControl = trainControl(method="none")), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:earth", unload=TRUE), error = function(e){print(e)})
  }
  
  ####################
  ###  Base Lines  ###   
  ####################
  
  #Optimal Classifier
  
  #Optimal class
  if (mymethod == "OptimalClass") { 
    preds <- factor(test$Class, levels=unique(test$Class))
  }
  #DreadFul Class
  if (mymethod == "PessimalClass") { 
    preds <- rep(FALSE, nrow(test))
  }
  
  #Majority class
  if (mymethod == "MajorityClass") { 
    preds <- factor(rep(majority(datos$Class),nrow(test)), levels=unique(test$Class))
  }
  
  #Minority class
  if (mymethod == "MinorityClass") { 
    minC <- which.min(as.vector(table(datos$Class)))
    minC_class <- dimnames(table(datos$Class))[[1]][minC]       
    preds <- factor(rep(minC_class,nrow(test)), levels=unique(test$Class))
  }
  
  #random Class
  
  if (mymethod == "RandomClass_A") { 
    #preds <- sample(unique(datos$Class),nrow(test), replace=T)
    preds <- sample(test$Class) 
    
  }
  
  if (mymethod == "RandomClass_B") { 
    #preds <- sample(unique(datos$Class),nrow(test), replace=T)
    preds <- sample(test$Class) 
    
  }
  
  if (mymethod == "RandomClass_C") { 
    #preds <- sample(unique(datos$Class),nrow(test), replace=T)
    preds <- sample(test$Class) 
    
  }
  
  if (mymethod == "RandomClass_D") { 
    #preds <- sample(unique(datos$Class),nrow(test), replace=T)
    preds <- sample(test$Class) 
    
  }
  
  if (mymethod == "RandomClass_E") { 
    #preds <- sample(unique(datos$Class),nrow(test), replace=T)
    preds <- sample(test$Class) 
    
  }
  
  if (mymethod == "RandomClass_F") { 
    #preds <- sample(unique(datos$Class),nrow(test), replace=T)
    preds <- sample(test$Class) 
    
  }
  
  if (mymethod == "RandomClass_G") { 
    #preds <- sample(unique(datos$Class),nrow(test), replace=T)
    preds <- sample(test$Class) 
    
  }
  
  if (mymethod == "RandomClass_H") { 
    #preds <- sample(unique(datos$Class),nrow(test), replace=T)
    preds <- sample(test$Class) 
    
  }
  
  if (mymethod == "RandomClass_I") { 
    #preds <- sample(unique(datos$Class),nrow(test), replace=T)
    preds <- sample(test$Class) 
    
  }
  
  if (mymethod == "RandomClass_J") { 
    #preds <- sample(unique(datos$Class),nrow(test), replace=T)
    preds <- sample(test$Class) 
    
  }
  
  ## Control Experiment
  
  if (mymethod == "gt90%_A") { 
    FAILS <- 1
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
    print("Salgo de gt90A")
  }
  
  if (mymethod == "gt90%_B") { 
    FAILS <- 1
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt90%_C") { 
    FAILS <- 1
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  ################
  
  if (mymethod == "gt80%_A") { 
    FAILS <- 2
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt80%_B") { 
    FAILS <- 2
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt80%_C") { 
    FAILS <- 2
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  
  ############
  
  
  if (mymethod == "gt70%_A") { 
    FAILS <- 3
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt70%_B") { 
    FAILS <- 3
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt70%_C") { 
    FAILS <- 3
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  
  ##############
  
  
  
  if (mymethod == "gt60%_A") { 
    FAILS <- 4
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt60%_B") { 
    FAILS <- 4
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt60%_C") { 
    FAILS <- 4
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  ##################
  
  
  
  if (mymethod == "gt50%_A") { 
    FAILS <- 5
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt50%_B") { 
    FAILS <- 5
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt50%_C") { 
    FAILS <- 5
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  ###################
  
  if (mymethod == "gt40%_A") { 
    FAILS <- 6
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt40%_B") { 
    FAILS <- 6
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt40%_C") { 
    FAILS <- 6
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  ############
  
  if (mymethod == "gt30%_A") { 
    FAILS <- 7
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt30%_B") { 
    FAILS <- 7
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt30%_C") { 
    FAILS <- 7
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  #############
  
  if (mymethod == "gt20%_A") { 
    FAILS <- 8
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt20%_B") { 
    FAILS <- 8
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt20%_C") { 
    FAILS <- 8
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  ###################
  
  
  if (mymethod == "gt10%_A") { 
    FAILS <- 9
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt10%_B") { 
    FAILS <- 9
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (mymethod == "gt10%_C") { 
    FAILS <- 9
    N <- round(FAILS*(nrow(test)/10))
    preds <- as.character(test$Class)
    s <- sample(length(preds),N)
    j <- 1
    for(i in s){
      preds[i] <- "FAIL"
      j<-j+1
    }
  }
  
  if (is.logical(ERROR)){
    if(ERROR == TRUE){
      print("------------ERROR")}
  }
  
  ################
  # Metodes Raul #
  ################


  
  if (mymethod == "bstSm") { 
    load_install("bst")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bstSm"), trace=TRUE, error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:bst", unload = TRUE), error = function(e){print(e)})
  }
  if (mymethod == "chaid") {
    load_install("CHAID")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "chaid"), trace=FALSE, error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:CHAID", unload = TRUE), error = function(e){print(e)})
  }
  if (mymethod == "deepboost") { 
    load_install("deepboost")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "deepboost"), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:deepboost", unload = TRUE), error = function(e){print(e)})
  }
  if (mymethod == "dda") { 
    load_install("sparsediscrim")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "dda"), trace=TRUE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:sparsediscrim", unload = TRUE), error = function(e){print(e)})
  }

  if (mymethod == "dwdPoly") {
    load_install("kerndwd")
	  ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "dwdPoly"), trace=FALSE, error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {print(e);return(rep(FALSE, nrow(test)))})
    }
	  tryCatch(detach("package:kerndwd", unload = TRUE), error = function(e){print(e)})
  }

  if (mymethod == "dwdRadial") { 
    load_install("kerndwd")
	  ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "dwdRadial"), trace=FALSE, error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {print(e);return(rep(FALSE, nrow(test)))})
    }
	  tryCatch(detach("package:kerndwd", unload = TRUE), error = function(e){print(e)})
  }

  # No peta, pero tarda moltissim
  if (mymethod == "xgbLinear") { 
    load_install("xgboost")
	ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "xgbLinear"), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
	tryCatch(detach("package:xgboost", unload = TRUE), error = function(e){print(e)})
  }

  if (mymethod == "elm") { 
    load_install("elmNN")
	ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "elm"), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
	tryCatch(detach("package:elmNN", unload = TRUE), error = function(e){print(e)})
  }
  if (mymethod == "FRBCS.CHI") { 
    load_install("frbs")
	ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "FRBCS.CHI"), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
	tryCatch(detach("package:frbs", unload = TRUE), error = function(e){print(e)})
  }

  if (mymethod == "GFS.GCCL") { 
	ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "GFS.GCCL"), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }
  if (mymethod == "gaussprLinear") { 
    load_install("kernlab")
	ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gaussprLinear"), error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
	tryCatch(detach("package:kernlab", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "gaussprPoly") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gaussprPoly"), error = function(e) {print(e);return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "gaussprRadial") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gaussprRadial"), error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "gamLoess") {
    load_install("gam")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gamLoess"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gam", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "bam") {
    load_install("mgcv")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bam"), trace=FALSE, error = function(e) {print(e);return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {print(e);return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:mgcv", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "gpls") {
    load_install("gpls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gpls"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:gpls", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "glmnet") {
    load_install("glmnet")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "glmnet"), trace=FALSE, error = function(e) {print(e);return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:glmnet", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "gbm_h2o") {
    load_install("h2o")
    h2o.init()
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gbm_h2o"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:h2o", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "hda") {
    load_install("hda")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "hda"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:hda", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "lda") {
    load_install("MASS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "lda"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:MASS", unload = TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "lda2") {
    load_install("MASS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "lda2"), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:MASS", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "logreg") {
    load_install("LogicReg")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "logreg"), trace=FALSE, error = function(e) {print(e);return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:LogicReg", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "Mlda") {
    load_install("HiDimDA")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "Mlda"), trace=FALSE, error = function(e) {print(e);return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:HiDimDA", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "manb") {
    load_install("bnclassify")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "manb"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:bnclassify", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "mlpML_3") {
    load_install("RSNNS")
    mlp_grid = expand.grid(layer1 = 3,
                           layer2 = 3,
                           layer3 = 3)
    
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc =  c("center", "scale"),method = "mlpML", tuneGrid = mlp_grid), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }

  if (mymethod == "mlpML") {
    load_install("RSNNS")
    
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc =  c("center", "scale"),method = "mlpML"), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "mlpML_2") {
    load_install("RSNNS")
    mlp_grid = expand.grid(layer1 = 3,
                           layer2 = 3,
                           layer3 = 0)    
    
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc =  c("center", "scale"),method = "mlpML", tuneGrid = mlp_grid), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }

  if (mymethod == "earth") {
    load_install("earth")
    
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "earth", trace=TRUE), error = function(e) {print(e);return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:earth", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "nb") {
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "nb"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }


  if (mymethod == "pam") {
    load_install("pamr")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "pam"), error = function(e) {print(e);return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:pamr", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "ORFlog") {
    load_install("obliqueRF")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ORFlog"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:obliqueRF", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "polr") {
    load_install("MASS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "polr", trControl = trainControl(method="probit")), trace=FALSE, error = function(e) {print(e);return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {print(e);return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:MASS", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "partDSA") {
    load_install("partDSA")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "partDSA"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:partDSA", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "PRIM") {
    load_install("supervisedPRIM")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "PRIM"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:supervisedPRIM", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "qda") {
    load_install("MASS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "qda"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:MASS", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "rFerns") {
    
    load_install("rFerns")
    
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rFerns"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:rFerns", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "extraTrees") {
    load_install("extraTrees")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "extraTrees"), trace=FALSE, error = function(e) {return(TRUE)})
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:extraTrees", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "rocc") {
    load_install("rocc")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rocc"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:rocc", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "rotationForest") {
    load_install("rotationForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rotationForest"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:rotationForest", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "xyf") {
    load_install("kohonen")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "xyf"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kohonen", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "CSimca") {
    load_install("rrcovHD")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "CSimca"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:rrcovHD", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "dnn") {
    load_install("deepnet")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "dnn"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:deepnet", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "svmBoundrangeString") {
    load_install("kernlab")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmBoundrangeString"), trace=FALSE, error = function(e) {print(e);return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {print(e);return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:kernlab", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "svmExpoString") {
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmExpoString"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }


  if (mymethod == "svmSpectrumString") {
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmSpectrumString"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
  }


  if (mymethod == "tan") {
    load_install("bnclassify")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "tan"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:bnclassify", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "evtree") {
    load_install("evtree")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "evtree"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:evtree", unload = TRUE), error = function(e){print(e)})
  }


  if (mymethod == "vbmpRadial") {
    load_install("vbmp")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "vbmpRadial"), trace=FALSE, error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:vbmp", unload = TRUE), error = function(e){print(e)})
  }

  if (mymethod == "nnet") {
    load_install("nnet")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "nnet"), error = function(e) {return(TRUE)})

    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:nnet", unload = TRUE), error = function(e){print(e)})
  }

# Model: Bayesian Generalized Linear Model
# Method: bayesglm
  if (mymethod == "bayesglm") { 
    load_install("arm")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bayesglm"), trace=TRUE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:arm", unload = TRUE), error = function(e){print(e)})
  }

# Model: Boosted Generalized Additive Model
# Method: gamboost
	if (mymethod == "gamboost") {
	  load_install("mboost")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gamboost"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
		tryCatch(detach("package:mboost", unload = TRUE), error = function(e){print(e)})
	}

# Model: Bayesian Additive Regression Trees
# Method: bartMachine
	if (mymethod == "bartMachine") {
	  load_install("bartMachine")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bartMachine"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
		tryCatch(detach("package:bartMachine", unload = TRUE), error = function(e){print(e)})
	}

# Model: Adaptive Mixture Discriminant Analysis
# Method: amdai
	if (mymethod == "amdai") {
	  load_install("adaptDA")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "amdai"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
		tryCatch(detach("package:adaptDA", unload = TRUE), error = function(e){print(e)})
	}

# Model: Bagged FDA using gCV Pruning
# Method: bagFDAGCV
	if (mymethod == "bagFDAGCV") {
	  load_install("earth")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bagFDAGCV"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
		tryCatch(detach("package:earth", unload = TRUE), error = function(e){print(e)})
	}

# Model: Boosted Linear Model
# Method: BstLm
	if (mymethod == "BstLm") {
            load_install("bst")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "BstLm"), trace=TRUE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:bst", unload = TRUE), error = function(e){print(e)})

	}

# Model: Boosted Tree
# Method: blackboost
	if (mymethod == "blackboost") {
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "blackboost"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
	}

# Model: Boosted Tree
# Method: bstTree
	if (mymethod == "bstTree") {
            load_install("bst")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "bstTree"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:bst", unload = TRUE), error = function(e){print(e)})
	}

# Model: Ensembles of Generalized Linear Models
# Method: randomGLM
	if (mymethod == "randomGLM") {
            load_install("randomGLM")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "randomGLM"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:randomGLM", unload = TRUE), error = function(e){print(e)})
	}

# Model: eXtreme Gradient Boosting
# Method: xgbDART
	if (mymethod == "xgbDART") {
            load_install("xgboost")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "xgbDART"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:xgboost", unload = TRUE), error = function(e){print(e)})
	}


# Model: Factor-Based Linear Discriminant Analysis
# Method: RFlda
	if (mymethod == "RFlda") {
            load_install("HiDimDA")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "RFlda"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:HiDimDA", unload = TRUE), error = function(e){print(e)})
	}

# Model: Fuzzy Rules Using the Structural Learning Algorithm on Vague Environment
# Method: SLAVE
	if (mymethod == "SLAVE") {
            load_install("frbs")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "SLAVE"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:frbs", unload = TRUE), error = function(e){print(e)})
	}

# Model: Fuzzy Rules with Weight Factor
# Method: FRBCS.W
	if (mymethod == "FRBCS.W") {
            load_install("frbs")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "FRBCS.W"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:frbs", unload = TRUE), error = function(e){print(e)})
	}

# Model: Generalized Additive Model using Splines
# Method: gam
	if (mymethod == "gam") {
            load_install("mgcv")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gam"), error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:mgcv", unload = TRUE), error = function(e){print(e)})
	}

# Model: Generalized Additive Model using Splines
# Method: gamSpline
	if (mymethod == "gamSpline") {
            load_install("gam")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "gamSpline"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:gam", unload = TRUE), error = function(e){print(e)})
	}

# Model: Generalized Linear Model with Stepwise Feature Selection
# Method: glmStepAIC
	if (mymethod == "glmStepAIC") {
            load_install("MASS")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "glmStepAIC"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:MASS", unload = TRUE), error = function(e){print(e)})
	}

# Model: Greedy Prototype Selection
# Method: protoclass
	if (mymethod == "protoclass") {
            load_install("proxy")
            load_install("protoclass")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "protoclass"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:protoclass", unload = TRUE), error = function(e){print(e)})
                tryCatch(detach("package:proxy", unload = TRUE), error = function(e){print(e)})
	}

# Model: High Dimensional Discriminant Analysis
# Method: hdda
	if (mymethod == "hdda") {
            load_install("HDclassif")
        
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "hdda"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:HDclassif", unload = TRUE), error = function(e){print(e)})
	}

# Model: High-Dimensional Regularized Discriminant Analysis
# Method: hdrda
	if (mymethod == "hdrda") {
            load_install("sparsediscrim")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "hdrda"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:sparsediscrim", unload = TRUE), error = function(e){print(e)})
	}
	

# Model: L2 Regularized Linear Support Vector Machines with Class Weights
# Method: svmLinearWeights2
	if (mymethod == "svmLinearWeights2") {
            load_install("LiblineaR")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmLinearWeights2"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:LiblineaR", unload = TRUE), error = function(e){print(e)})
	}

# Model: L2 Regularized Support Vector Machine (dual) with Linear Kernel
# Method: svmLinear3
	if (mymethod == "svmLinear3") {
            load_install("LiblineaR")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmLinear3"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:LiblineaR", unload = TRUE), error = function(e){print(e)})
	}

# Model: Least Squares Support Vector Machine
# Method: lssvmLinear
	if (mymethod == "lssvmLinear") {
            load_install("kernlab")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "lssvmLinear"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:kernlab", unload = TRUE), error = function(e){print(e)})
	}

# Model: Least Squares Support Vector Machine with Polynomial Kernel
# Method: lssvmPoly
	if (mymethod == "lssvmPoly") {
            load_install("kernlab")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "lssvmPoly"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {print(e); return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:kernlab", unload = TRUE), error = function(e){print(e)})
	}

# Model: Least Squares Support Vector Machine with Radial Basis Function Kernel
# Method: lssvmRadial
	if (mymethod == "lssvmRadial") {
            load_install("kernlab")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "lssvmRadial"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:kernlab", unload = TRUE), error = function(e){print(e)})
	}

# Model: Linear Discriminant Analysis with Stepwise Feature Selection
# Method: stepLDA
	if (mymethod == "stepLDA") {
            load_install("klaR")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "stepLDA"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:klaR", unload = TRUE), error = function(e){print(e)})
	}

# Model: Linear Distance Weighted Discrimination
# Method: dwdLinear
	if (mymethod == "dwdLinear") {
            load_install("kerndwd")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "dwdLinear"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:kerndwd", unload = TRUE), error = function(e){print(e)})
	}

# Model: Localized Linear Discriminant Analysis
# Method: loclda
	if (mymethod == "loclda") {
            load_install("klaR")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "loclda"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:klaR", unload = TRUE), error = function(e){print(e)})
	}

# Model: Multi-Layer Perceptron
# Method: mlpWeightDecay
	if (mymethod == "mlpWeightDecay") {
            load_install("RSNNS")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
	}
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e06
  if (mymethod == "mlpWeightDecay_1e06") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay", tuneGrid=data.frame(size=5, decay = 1e-06)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e05
  if (mymethod == "mlpWeightDecay_1e05") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay", tuneGrid=data.frame(size=5, decay = 1e-05)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e04
  if (mymethod == "mlpWeightDecay_1e04") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay", tuneGrid=data.frame(size=5, decay = 1e-04)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e03
  if (mymethod == "mlpWeightDecay_1e03") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay", tuneGrid=data.frame(size=5, decay = 1e-03)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e02
  if (mymethod == "mlpWeightDecay_1e02") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay", tuneGrid=data.frame(size=5, decay = 1e-02)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e01
  if (mymethod == "mlpWeightDecay_1e01") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay", tuneGrid=data.frame(size=5, decay = 1e-01)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 0
  if (mymethod == "mlpWeightDecay_0") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay", tuneGrid=data.frame(size=5, decay = 0)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 0.3
  if (mymethod == "mlpWeightDecay_03") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay", tuneGrid=data.frame(size=5, decay = 0.3)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 0.6
  if (mymethod == "mlpWeightDecay_06") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay", tuneGrid=data.frame(size=5, decay = 0.6)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1
  if (mymethod == "mlpWeightDecay_1") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpWeightDecay", tuneGrid=data.frame(size=5, decay = 1)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }

# Model: Multi-Layer Perceptron| multiple layers
# Method: mlpWeightDecayML
	if (mymethod == "mlpWeightDecayML") {
            load_install("RSNNS")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML",tuneGrid=data.frame(layer1=3, layer2=3, layer3=3, decay = 1e-05)), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
	}

  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e06
  if (mymethod == "mlpWeightDecayML_1e06") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=3, decay = 1e-06)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e05
  if (mymethod == "mlpWeightDecayML_2_1e05") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=0, decay = 1e-05)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  if (mymethod == "mlpWeightDecayML_3_1e05") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=3, decay = 1e-05)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e04
  if (mymethod == "mlpWeightDecayML_1e04") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=3,decay = 1e-04)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e03
  if (mymethod == "mlpWeightDecayML_1e03") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=3, decay = 1e-03)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e02
  if (mymethod == "mlpWeightDecayML_1e02") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=3, decay = 1e-02)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1e01
  if (mymethod == "mlpWeightDecayML_1e01") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=3, decay = 1e-01)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 0
  if (mymethod == "mlpWeightDecayML_0") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=3, decay = 0)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 0.3
  if (mymethod == "mlpWeightDecayML_03") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=3, decay = 0.3)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 0.6
  if (mymethod == "mlpWeightDecayML_06") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=3, decay = 0.6)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
  # Model: Multi-Layer Perceptron
  # Method: mlpWeightDecay with decay 1
  if (mymethod == "mlpWeightDecayML_1") {
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc=c("center", "scale"), method = "mlpWeightDecayML", tuneGrid=data.frame(layer1=3, layer2=3, layer3=3, decay = 1)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  
# Model: Multi-Step Adaptive MCP-Net
# Method: msaenet
	if (mymethod == "msaenet") {
            load_install("msaenet")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "msaenet"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:msaenet", unload = TRUE), error = function(e){print(e)})
	}

# Model: Multilayer Perceptron Network with Dropout
# Method: mlpKerasDropoutCost
	if (mymethod == "mlpKerasDropoutCost") {
            load_install("keras")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpKerasDropoutCost"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:keras", unload = TRUE), error = function(e){print(e)})
	}

# Model: Multilayer Perceptron Network with Weight Decay
# Method: mlpKerasDecay
	if (mymethod == "keras") {
            load_install("msaenet")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpKerasDecay"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:keras", unload = TRUE), error = function(e){print(e)})
	}

# Model: Multilayer Perceptron Network with Weight Decay
# Method: mlpKerasDecayCost
	if (mymethod == "mlpKerasDecayCost") {
            load_install("keras")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mlpKerasDecayCost"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:keras", unload = TRUE), error = function(e){print(e)})
	}

# Model: Naive Bayes Classifier with Attribute Weighting
# Method: awnb
	if (mymethod == "awnb") {
            load_install("bnclassify")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "awnb"), error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:bnclassify", unload = TRUE), error = function(e){print(e)})
	}

# Model: Neural Network
# Method: mxnet
	if (mymethod == "mxnet") {
            load_install("mxnet")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mxnet"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:mxnet", unload = TRUE), error = function(e){print(e)})
	}

# Model: Neural Network
# Method: mxnetAdam
	if (mymethod == "mxnetAdam") {
            load_install("mxnet")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "mxnetAdam"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:mxnet", unload = TRUE), error = function(e){print(e)})
	}

# Model: Oblique Random Forest
# Method: ORFpls
	if (mymethod == "ORFpls") {
            load_install("obliqueRF")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ORFpls"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:obliqueRF", unload = TRUE), error = function(e){print(e)})
	}

# Model: Oblique Random Forest
# Method: ORFridge
	if (mymethod == "ORFridge") {
            load_install("obliqueRF")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ORFridge"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:obliqueRF", unload = TRUE), error = function(e){print(e)})
	}

# Model: Oblique Random Forest
# Method: ORFsvm
	if (mymethod == "ORFsvm") {
            load_install("obliqueRF")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ORFsvm"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:obliqueRF", unload = TRUE), error = function(e){print(e)})
	}

# Model: Optimal Weighted Nearest Neighbor Classifier
# Method: ownn
	if (mymethod == "ownn") {
            load_install("snn")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ownn"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:snn", unload = TRUE), error = function(e){print(e)})
	}
  
  if (mymethod == "ownn_k5") {
    load_install("snn")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ownn", tuneGrid=data.frame(K = 5)), trace=FALSE, error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:snn", unload = TRUE), error = function(e){print(e)})
  }

# Model: Partial Least Squares
# Method: kernelpls
	if (mymethod == "kernelpls") {
            load_install("pls")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "kernelpls"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:pls", unload = TRUE), error = function(e){print(e)})
	}

# Model: Partial Least Squares
# Method: kernelpls
  if (mymethod == "kernelpls_ncomp1") {
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "kernelpls" ,tuneGrid=data.frame(ncomp = 1),  trControl = trainControl(method="none")), error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:pls", unload = TRUE), error = function(e){print(e)})
  }
  
# Model: Partial Least Squares
# Method: kernelpls  
  if (mymethod == "kernelpls_ncomp2") {
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "kernelpls" ,tuneGrid=data.frame(ncomp = 2),  trControl = trainControl(method="none")), error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:pls", unload = TRUE), error = function(e){print(e)})
  }
  
# Model: Partial Least Squares
# Method: widekernelpls
	if (mymethod == "widekernelpls") {
            load_install("pls")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "widekernelpls"), error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:pls", unload = TRUE), error = function(e){print(e)})
	}

  # Model: Partial Least Squares
  # Method: widekernelpls
  if (mymethod == "widekernelpls_ncomp1") {
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "widekernelpls",tuneGrid=data.frame(ncomp = 1),  trControl = trainControl(method="none")), error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:pls", unload = TRUE), error = function(e){print(e)})
  }
  
# Model: Partial Least Squares
# Method: widekernelpls
  if (mymethod == "widekernelpls_ncomp2") {
    load_install("pls")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "widekernelpls",tuneGrid=data.frame(ncomp = 2),  trControl = trainControl(method="none")), error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:pls", unload = TRUE), error = function(e){print(e)})
  }
  
# Model: Partial Least Squares Generalized Linear Models
# Method: plsRglm
	if (mymethod == "plsRglm") {
            load_install("plsRglm")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "plsRglm"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:plsRglm", unload = TRUE), error = function(e){print(e)})
	}

# Model: Penalized Discriminant Analysis
# Method: pda
	if (mymethod == "pda") {
            load_install("mda")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "pda"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:mda", unload = TRUE), error = function(e){print(e)})
	}

# Model: Penalized Discriminant Analysis
# Method: pda2
	if (mymethod == "pda2") {
            load_install("mda")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "pda2"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:mda", unload = TRUE), error = function(e){print(e)})
	}

# Model: Penalized Linear Discriminant Analysis
# Method: PenalizedLDA
	if (mymethod == "PenalizedLDA") {
            load_install("penalizedLDA")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "PenalizedLDA"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {print(e);return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:penalizedLDA", unload = TRUE), error = function(e){print(e)})
	}

# Model: Penalized Multinomial Regression
# Method: multinom
	if (mymethod == "multinom") {
            load_install("nnet")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "multinom"), error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:nnet", unload = TRUE), error = function(e){print(e)})
	}

# Model: Penalized Ordinal Regression
# Method: ordinalNet
	if (mymethod == "ordinalNet") {
            load_install("ordinalNet")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "ordinalNet"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:ordinalNet", unload = TRUE), error = function(e){print(e)})
	}

# Model: Quadratic Discriminant Analysis with Stepwise Feature Selection
# Method: stepQDA
	if (mymethod == "stepQDA") {
            load_install("klaR")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "stepQDA"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:klaR", unload = TRUE), error = function(e){print(e)})
	}

# Model: Random Forest Rule-Based Model
# Method: rfRules
	if (mymethod == "rfRules") {
            load_install("inTrees")
            load_install("randomForest")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rfRules"), error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
    tryCatch(detach("package:randomForest", unload = TRUE), error = function(e){print(e)})
    tryCatch(detach("package:inTrees", unload = TRUE), error = function(e){print(e)})
	}

  # Model: Random Forest Rule-Based Model
  # Method: rfRules
  if (mymethod == "rfRules_mtry16") {
    load_install("inTrees")
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rfRules", tuneGrid=data.frame(mtry = 16, maxdepth=2),  trControl = trainControl(method="none")), error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload = TRUE), error = function(e){print(e)})
    tryCatch(detach("package:inTrees", unload = TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "rfRules_mtry64") {
    load_install("inTrees")
    load_install("randomForest")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rfRules", tuneGrid=data.frame(mtry = 64, maxdepth=2),  trControl = trainControl(method="none")), error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    } else {
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:randomForest", unload = TRUE), error = function(e){print(e)})
    tryCatch(detach("package:inTrees", unload = TRUE), error = function(e){print(e)})
  }
  
# Model: Regularized Discriminant Analysis
# Method: rda
	if (mymethod == "rda") {
            load_install("klaR")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rda"), trace=TRUE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:klaR", unload = TRUE), error = function(e){print(e)})
	}

# Model: Robust Linear Discriminant Analysis
# Method: Linda
	if (mymethod == "Linda") {
            load_install("rrcov")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "Linda"), error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:rrcov", unload = TRUE), error = function(e){print(e)})
	}

# Model: Robust Mixture Discriminant Analysis
# Method: rmda
	if (mymethod == "rmda") {
            load_install("robustDA")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rmda"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:robustDA", unload = TRUE), error = function(e){print(e)})
	}

# Method: QdaCov
	if (mymethod == "QdaCov") {
            load_install("rrcov")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "QdaCov"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:rrcov", unload = TRUE), error = function(e){print(e)})
	}

# Model: Robust SIMCA
# Method: RSimca
	if (mymethod == "RSimca") {
            load_install("rrcovHD")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "RSimca"), error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:rrcovHD", unload = TRUE), error = function(e){print(e)})
	}

# Model: Rotation Forest
# Method: rotationForestCp
	if (mymethod == "rotationForestCp") {
            load_install("rotationForest")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rotationForestCp"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:rotationForest", unload = TRUE), error = function(e){print(e)})
	}

# Model: Semi-Naive Structure Learner Wrapper
# Method: nbSearch
	if (mymethod == "nbSearch") {
            load_install("bnclassify")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "nbSearch"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:bnclassify", unload = TRUE), error = function(e){print(e)})
	}


# Model: Sparse Distance Weighted Discrimination
# Method: sdwd
	if (mymethod == "sdwd") {
            load_install("sdwd")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "sdwd"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:sdwd", unload = TRUE), error = function(e){print(e)})
	}

# Model: Sparse Linear Discriminant Analysis
# Method: sparseLDA
	if (mymethod == "sparseLDA") {
            load_install("sparseLDA")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "sparseLDA"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:sparseLDA", unload = TRUE), error = function(e){print(e)})
	}

# Model: Sparse Mixture Discriminant Analysis
# Method: smda
	if (mymethod == "smda") {
            load_install("sparseLDA")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "smda"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:sparseLDA", unload = TRUE), error = function(e){print(e)})
	}

# Model: Sparse Partial Least Squares
# Method: spls
	if (mymethod == "spls") {
            load_install("spls")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "spls"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:spls", unload = TRUE), error = function(e){print(e)})
	}

# Model: Stabilized Linear Discriminant Analysis
# Method: slda
	if (mymethod == "slda") {
            load_install("ipred")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "slda"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:ipred", unload = TRUE), error = function(e){print(e)})
	}

# Model: Support Vector Machines with Class Weights
# Method: svmRadialWeights
	if (mymethod == "svmRadialWeights") {
            load_install("kernlab")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "svmRadialWeights"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:kernlab", unload = TRUE), error = function(e){print(e)})
	}

# Model: Tree Augmented Naive Bayes Classifier Structure Learner Wrapper
# Method: tanSearch
	if (mymethod == "tanSearch") {
            load_install("bnclassify")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "tanSearch"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:bnclassify", unload = TRUE), error = function(e){print(e)})
	}

# Model: Tree Augmented Naive Bayes Classifier with Attribute Weighting
# Method: awtan
	if (mymethod == "awtan") {
            load_install("bnclassify")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "awtan"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:bnclassify", unload = TRUE), error = function(e){print(e)})
	}

# Model: Tree-Based Ensembles
# Method: nodeHarvest
	if (mymethod == "nodeHarvest") {
            load_install("nodeHarvest")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "nodeHarvest"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:nodeHarvest", unload = TRUE), error = function(e){print(e)})
	}

# Model: Weighted Subspace Random Forest
# Method: wsrf
	if (mymethod == "wsrf") {
            load_install("wsrf")
		ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "wsrf"), trace=FALSE, error = function(e) {print(e); return(TRUE)})

		if (is.logical(ERROR)){
			preds <- rep(FALSE, nrow(test))
		} else {
			preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
		}
                tryCatch(detach("package:wsrf", unload = TRUE), error = function(e){print(e)})
	}

# Model: Boosted Generalized Linear Model
# Method: glmboost
  if (mymethod == "glmboost") { 
            load_install("mboost")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "glmboost"), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
        tryCatch(detach("package:mboost", unload = TRUE), error = function(e){print(e)})
  }

  # Model: Penalized Logistic Regression
  # Method: plr
  if (mymethod == "plr") { 
    load_install("stepPlr")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "plr"), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:stepPlr", unload = TRUE), error = function(e){print(e)})
  }

  # Model: Penalized Logistic Regression
  # Method: plr
  if (mymethod == "plr_bic_1e03") { 
    load_install("stepPlr")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "plr", tuneGrid=data.frame(cp = "bic", lambda = 1e-03)), trace=FALSE, error = function(e) {print(e);return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:stepPlr", unload = TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "plr_bic_1") { 
    load_install("stepPlr")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "plr", tuneGrid=data.frame(cp = "bic", lambda = 1)), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:stepPlr", unload = TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "plr_aic_1e03") { 
    load_install("stepPlr")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "plr", tuneGrid=data.frame(cp = "aic", lambda = 1e-03)), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:stepPlr", unload = TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "plr_aic_1e01") { 
    load_install("stepPlr")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "plr", tuneGrid=data.frame(cp = "aic", lambda = 1e-01)), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:stepPlr", unload = TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "plr_aic_1") { 
    load_install("stepPlr")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "plr", tuneGrid=data.frame(cp = "aic", lambda = 1)), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:stepPlr", unload = TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "regLogistic") { 
    load_install("LiblineaR")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "plr", tuneGrid=data.frame(cp = "aic", lambda = 1)), trace=FALSE, error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:LiblineaR", unload = TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "Bagging"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- Bagging(Class ~., data = train), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "Stacking"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- Stacking(Class ~., data = train), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
  if (mymethod == "M5P"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- RWeka::M5P(Class ~., data = train), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  
 
  if (mymethod == "M5Rules"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    ERROR <-  tryCatch(model <- RWeka::M5Rules(Class ~., data = train), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "LBR"){
    load_install("RWeka")
    preProc <- preProcess(train, method=c("center", "scale"))
    train <- predict(preProc, train)
    test <- predict(preProc, test)
    
    
    LBR <- make_Weka_classifier("weka/classifiers/lazy/LBR",
                                c('LBR', 'Weka_lazy'),
                                init=make_Weka_package_loader("lazyBayesianRules"))
    
    ERROR <-  tryCatch(model <- LBR(Class ~., data = train), error = function(e) {return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RWeka", unload=TRUE), error = function(e){print(e)})
  }
  if (mymethod == "rbfDDA") { 
    load_install("RSNNS")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "rbfDDA"), error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:RSNNS", unload = TRUE), error = function(e){print(e)})
  }
  if (mymethod == "snn") { 
    load_install("snn")
    ERROR <-  tryCatch(model <- caret::train(Class ~ ., data = train, preProc = c("center", "scale"), method = "snn"), error = function(e) {print(e); return(TRUE)})
    
    if (is.logical(ERROR)){
      preds <- rep(FALSE, nrow(test))
    }else{
      preds<- tryCatch(predict(model, newdata = test), error = function(e) {return(rep(FALSE, nrow(test)))})
    }
    tryCatch(detach("package:snn", unload = TRUE), error = function(e){print(e)})
  }
  
  #save(model, file=model_path)
  
  return(preds)
}

