init_2D <- function() {
  source("2_family_identification/utils.R")
  source("common/common.R")
  library(mlbench)
  library(ggplot2)
  library(gridExtra)
  library(entropy)
  
  options( java.parameters = "-Xmx6g" )
  .lib<- c("ggplot2","plyr","dplyr","RWeka", "sampling","gbm", "rpart", "e1071","randomForest","foreach","MASS","mlbench",
           "rrlda","C50", "MASS","RSNNS", "class", "kernlab","party", "sda", "rrcov", "robustbase", "mda", 
           "ada", "caTools", "adabag","ipred", "bst", "randomForest", "RRF", "pls", "KODAMA","caret", "party")
  
  lapply(.lib, require, character.only=TRUE)
  
  models <- c("DT", "NB", "SVM", 
              "LR", "NN", "RBF", "RF")
  
  ds <- artificial_dataset()
  train <- ds[["train"]]
  uniform_test <- ds[["test"]]
  
  out <- list("train"=train, "test"=uniform_test, "models"=models)
  out
}

open_plot <- function(file, height, width) {
  if (grepl(".pdf", file, ignore.case = TRUE)) {
    pdf(file, width, height)
  }
  else if (grepl(".eps", file, ignore.case = TRUE)) {
    postscript(file, width, height, horizontal=FALSE)
  }
  else if (grepl(".png", file, ignore.case = TRUE)) {
    png(file, width, height)
  }
  else {
    print(paste("Non-recoginized file format:", file))
  }
}

artificial_dataset <- function() {
  
  set.seed(3)
  
  #Roig
  class1 <- data.frame(x.1 = runif(100, 0.00, 1.00),
                       x.2 = runif(100, 0.75, 1.00),
                       classes = "1") 
  
  set.seed(3)
  #verd
  class2 <- data.frame(x.1 = runif(100, 0.75, 1.00),
                       x.2 = runif(100, 0.00, 1.00),
                       classes = "2") 
  
  set.seed(3)
  # blau
  class3 <- data.frame(x.1 = runif(50, 0.40, 0.65),
                       x.2 = runif(50, 0.60, 0.80),
                       classes = "3") 
  
  set.seed(3)
  #morat
  class4 <- data.frame(x.1 = runif(50, 0.60, 0.75),
                       x.2 = runif(50, 0.50, 0.80),
                       classes = "4") 
  
  dt <- class1
  dt <- rbind(dt, class2)
  dt <- rbind(dt, class3)
  dt <- rbind(dt, class4)
  
  ds <- list()
  ds[["train"]] <- data.frame(x.1 = dt$x.1, x.2 = dt$x.2, classes = dt$classes)
  
  
  set.seed(3)
  ds[["test"]]  <- data.frame(x.1 = runif(10000,0,1),
                              x.2 = runif(10000,0,1),
                              classes = NA)
  
  set.seed(4)
  ds[["test2"]]  <- data.frame(x.1 = runif(10000,0,1),
                               x.2 = runif(10000,0,1),
                               classes = NA)
  
  ds
}



add_plot <- function(plot, args) {
  if (missing(args) || length(args) == 0) {
    args <- list("plot_list"=list(), "plot_idx"=1)
  }
  
  args$plot_list[[args$plot_idx]] <- plot
  args$plot_idx <- args$plot_idx + 1
  
  args
}

dataset_plot <- function(df, title="") {
  plot <- ggplot(df, aes(x.1,x.2, colour = classes)) +geom_point() + 
    ggtitle(title) + theme_minimal() + scale_color_discrete(guide = FALSE) + 
    xlim(0.0, 1.0) + ylim(0.0, 1.0)
  plot
}

entropy_map_plot <- function(data, title="\nEntropy map") {
  #plot_df <- cbind(test[,1:ncol(test)-1], entropy_df)
  ent_plot <- ggplot(data, aes(x.1, x.2, color=label_entropy)) + 
    geom_point() +       ggtitle(title) +
    scale_color_gradient(low="yellow", high="red") + theme_minimal() +
    theme(legend.position="none")
}
train_test <- function(p.method, p.train, p.test) {
  # p.train <- train
  # p.test <- test
  # p.method <- "Random Forest (rf)"
  library(e1071)
  library(nnet)
  require(randomForest)
  library(DMwR)
  
  if (p.method == "DT") {
    model <- C5.0(classes ~., data = p.train)
    preds <- predict(model, newdata = p.test)
  }
  else if (p.method == "NB") {
    model <- naiveBayes(classes ~., data = p.train)
    preds <- predict(model, newdata = p.test)
  }
  else if (p.method == "SVM") {
    model <- caret::train(classes ~ ., data = p.train, method = "svmRadialCost", trControl = trainControl(method="none"))
    preds<- predict(model, newdata = p.test)
  }
  else if (p.method == "LR") {
    model <- multinom(classes ~.,family=multinomial,data=p.train)
    preds<- predict(model, newdata = p.test)
  }
  else if (p.method == "NN") {
    preds <-kNN(classes ~., p.train, p.test, norm = F, k = 11)
  }
  else if (p.method == "RBF") {
    model <- caret::train(classes ~ ., data = p.train, method = "mlp", tuneGrid=data.frame(size=5), 
                          trControl = trainControl(method="none"))
    preds<- predict(model, newdata = p.test)
  }
  else if (p.method == "RF") {
   
    model <- caret::train(classes ~ ., data = p.train, method = "rf",tuneGrid=data.frame(mtry = 1)
                          ,  trControl = trainControl(method="none"))
    preds<- predict(model, newdata = p.test)
  }
  
  preds
}

surrogate_labels_2D <- function(train, test, models) {
  preds <- list()
  i_sm <- 0
  for (sm in models) {
    print(sprintf("Labeling test set. Surrogate %s", sm))
    i_sm <- i_sm + 1
    set.seed(3)
    preds[[sm]] <- train_test(sm, train, test) 
  }
  preds
}



plot_dataset_models <- function(out_path) {
  library(mlbench)
  
  #options( java.parameters = "-Xmx6g" )
  .lib<- c("ggplot2","plyr","dplyr","RWeka", "sampling","gbm", "rpart", "e1071","randomForest","foreach","MASS","mlbench",
           "rrlda","C50", "MASS","RSNNS", "class", "kernlab","party", "sda", "rrcov", "robustbase", "mda", 
           "ada", "caTools", "adabag","ipred", "bst", "randomForest", "RRF", "pls", "KODAMA","caret", "party")
  
  .inst <- .lib %in% installed.packages()
  if (length(.lib[!.inst])>0) install.packages(.lib[!.inst], repos=c("http://rstudio.org/_packages", "http://cran.rstudio.com")) 
  lapply(.lib, require, character.only=TRUE)
  
  library(ggplot2)
  
  init <- init_2D()
  
  train <- init$train
  test <- init$test
  
  orig <- ggplot(train, aes(x.1,x.2, colour = classes)) +geom_point() + 
    ggtitle("Original Data") + theme_minimal() + scale_color_discrete(guide = FALSE)
  
  #test_plot <- ggplot(test, aes(x.1,x.2)) +geom_point() + 
  #  ggtitle("Original Test") + theme_minimal() + scale_color_discrete(guide = FALSE)
  
  # 
  # train <- dt.train
  # test <- dt.test  
  
    #DT
    model <- C5.0(classes ~., data = train)
    preds <- predict(model, newdata = test)
    test$classes <- preds
    dt <- ggplot(test, aes(x.1,x.2, colour = classes)) + geom_point() + 
      ggtitle("Decision Tree (c5.0)") + theme_minimal() + scale_color_discrete(guide = FALSE)
    
    
    #NaiveBayes
    library(e1071)
    model <- naiveBayes(classes ~., data = train)
    preds <- predict(model, newdata = test)
    
    test$classes <- preds
    nb <- ggplot(test, aes(x.1,x.2, colour = classes)) +geom_point() + 
      ggtitle("Naive Bayes (nb)") + theme_minimal() + scale_color_discrete(guide = FALSE)
  
    #linear SVM
    library(e1071)
    model <- caret::train(classes ~ ., data = train, method = "svmRadialCost", trControl = trainControl(method="none"))
    preds<- predict(model, newdata = test)
    test$classes <- preds
    svm <- ggplot(test, aes(x.1,x.2, colour = classes)) +geom_point() + 
      ggtitle("SVM Radial (svmRadialCost)") + theme_minimal() + scale_color_discrete(guide = FALSE)
    
    
    #Log Regression
    library(nnet)
    model <- multinom(classes ~.,family=multinomial,data=train)
    preds<- predict(model, newdata = test)
    test$classes <- preds
    lr <- ggplot(test, aes(x.1,x.2, colour = classes)) +geom_point()+ 
      ggtitle("Logistic Regression (LR)") + theme_minimal() + scale_color_discrete(guide = FALSE)
    
   
    #Knn
    #install.packages("DMwR")
    library(DMwR)
    predsknn <-kNN(classes ~., train, test, norm = F, k = 11)
    test$classes <- predsknn
    knn <- ggplot(test, aes(x.1,x.2, colour = classes)) +geom_point()+ 
      #ggtitle("5???Nearest Neighbors (knn)") + theme_minimal() + scale_color_discrete(guide = FALSE)
      ggtitle("11-Nearest Neighbors (knn)") + theme_minimal() + scale_color_discrete(guide = FALSE)
    
    
    #NN
    model <- caret::train(classes ~ ., data = train, method = "mlp", tuneGrid=data.frame(size=5), 
                          trControl = trainControl(method="none"))
    predsNN<- predict(model, newdata = test)
    test$classes <- predsNN
    nn <- ggplot(test, aes(x.1,x.2, colour = classes)) +geom_point() + 
      ggtitle("Radial Basis Network (rbf)") + theme_minimal() + scale_color_discrete(guide = FALSE)
    
    
    #RF
    require(randomForest)
    model <- caret::train(classes ~ ., data = train, method = "rf",tuneGrid=data.frame(mtry = 1)
                          ,  trControl = trainControl(method="none"))
    predsrf<- predict(model, newdata = test)
    test$classes <- predsrf
    rf <- ggplot(test, aes(x.1,x.2, colour = classes)) + geom_point() + 
      ggtitle("Random Forest (rf)") + theme_minimal() + scale_color_discrete(guide = FALSE)
    
    
    library(gridExtra)
    g <- grid.arrange(orig, dt, nb,svm,lr,knn,nn,rf, nrow = 2)
    
    out_path <- correct_path(out_path)
    open_plot(sprintf("%smodel_boundaries.png", out_path), 500, 900)
    # open_plot/png("model_boundaries.png", height=500, width=900)
    grid.arrange(orig, dt, nb,svm,lr,knn,nn,rf, nrow = 2)
    dev.off()
}

plot_surrogate_entropy_kappa <- function(out_path="2_family_identification/files/plots/") {
  init <- init_2D()
  
  orig_plot <- dataset_plot(init$train, "\nOriginal Data")
  
  i_om <- 0
  for (om in init$models) {
    print(sprintf("Oracle: %s", om))
    
    plots <- add_plot(orig_plot)  
    i_om <- i_om + 1
    
    train <- init$train
    test <- init$test
    
    set.seed(3)
    
    test$classes <- train_test(om, train, test)
    
    test_plot <- dataset_plot(test, sprintf("Surrogate dataset\nlabeled by oracle (%s)", om))
    plots <- add_plot(test_plot, plots)
    
    surrogate_metadata <- list()
    surrogate_metadata$predictions <- surrogate_labels_2D(test, test, init$models) 
    surrogate_metadata$entropy <- dataset_entropy(surrogate_metadata$predictions, 
                                                  as.factor(unique(train$classes)))
    
    
    ent_plot <- entropy_map_plot(cbind(test[,1:ncol(test)-1], surrogate_metadata$entropy))  
    plots <- add_plot(ent_plot, plots)
    
    tr_idx <- split_dataset_entropy(surrogate_metadata$entropy, criterion="train-boundaries")
    s_train <- test[tr_idx,]
    s_test <- test[!tr_idx,]
    
    plots <- add_plot(dataset_plot(s_train, "\nSurrogate Traning"), plots)
    plots <- add_plot(dataset_plot(s_test, "\nSurrogate Test"), plots)
    
    s_test_metadata = list()
    s_test_metadata$predictions <- surrogate_labels_2D(s_train,
                                                    s_test, init$models) 
    
    s_test_metadata$kappa <- metafeat_kappa(s_test$classes, s_test_metadata$predictions, init$models)
    test2 <- s_test
    i_sm <- 1
    for (sm in init$models) {
      i_sm <- i_sm + 1
      
      test2$classes <- s_test_metadata$predictions[[sm]]
      kappa <- s_test_metadata$kappa[[sm]]
      
      plots <- add_plot(dataset_plot(test2, sprintf("Oracle %s,\nSurrogate %s", om, sm)) + 
                          geom_text(label = sprintf("kappa = %s", format(kappa, digits=3)), 
                                    x = 0.5, y = 0.9, colour = "black"), plots)
    }
    
    if (substr(out_path,start = nchar(out_path), stop=nchar(out_path)) != "/") {
      out_path <- paste(out_path, "/", sep="")
    }
    out_path <- correct_path(out_path)
    open_plot(sprintf("%sentropy_kappa_%s.png", out_path, om), 800, 1300)
    grid.arrange(grobs = plots$plot_list, ncol = 5)
    dev.off()
  }
}

plot_surrogate_kappa <- function(out_path="2_family_identification/files/plots/") {
  init <- init_2D()
  
  orig_plot <- dataset_plot(init$train, "\nOriginal Data")
  
  i_om <- 0
  for (om in init$models) {
    #om <- "DT"
    print(sprintf("Oracle: %s", om))
    
    plots <- add_plot(orig_plot)  
    i_om <- i_om + 1
    
    train <- init$train
    test <- init$test
    
    set.seed(3)
    
    test$classes <- train_test(om, train, test)
    
    test_plot <- dataset_plot(test, sprintf("Surrogate dataset\nlabeled by oracle (%s)", om))
    plots <- add_plot(test_plot, plots)
    
    surrogate_metadata <- list()
    surrogate_metadata$predictions <- surrogate_labels_2D(test, test, init$models) 
    surrogate_metadata$kappa <- metafeat_kappa(test$classes, surrogate_metadata$predictions, init$models)
    surrogate_metadata$entropy <- dataset_entropy(surrogate_metadata$predictions, 
                                                  as.factor(unique(train$classes)))
    
    
    ent_plot <- entropy_map_plot(cbind(test[,1:ncol(test)-1], surrogate_metadata$entropy))  
    plots <- add_plot(ent_plot, plots)
    
    i_sm <- 1
    test2 <- init$test
    for (sm in init$models) {
      i_sm <- i_sm + 1
      
      test2$classes <- surrogate_metadata$predictions[[sm]]
      kappa <- surrogate_metadata$kappa[[sm]]
      
      plots <- add_plot(dataset_plot(test2, sprintf("Oracle %s,\nSurrogate %s", om, sm)) + 
                          geom_text(label = sprintf("kappa = %s", format(kappa, digits=3)), 
                                    x = 0.5, y = 0.9, colour = "black"), plots)
    }
    
    if (substr(out_path,start = nchar(out_path), stop=nchar(out_path)) != "/") {
      out_path <- paste(out_path, "/", sep="")
    }
    out_path <- correct_path(out_path)
    open_plot(sprintf("%skappa_%s.png", out_path, om), 600, 1300)
    grid.arrange(grobs = plots$plot_list, ncol = 5)
    dev.off()
  }
}

# Example calls:
#
# plot_dataset_models("2_family_identification/files/plots/")
# plot_surrogate_kappa("2_family_identification/files/plots/")
# plot_surrogate_entropy_kappa("2_family_identification/files/plots")
