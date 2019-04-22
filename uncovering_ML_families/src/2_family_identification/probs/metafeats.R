mf.path_base <- "files/test/"
mf.ds_path <- "datasets/extension_caepia/"

metafeats <- function(mf.path_base, mf.ds_path) {
  
  source("src/2_family_identification/utils/utils.R")
  
  mf.path_base <- correct_path(mf.path_base)
  mf.ds_path <- correct_path(mf.ds_path)
  mf.preds_path <- correct_path(paste(mf.path_base, "predictions", sep=""))
  mf.analytics_path <- correct_path(paste(mf.path_base, "analytics", sep=""))
  
  mf.init <- init_probs(mf.path_base, mf.ds_path)
  
  mf.datasets <- mf.init$datasets#[1:10]
  
  mf.models <- mf.init$models
  
  mf.nds = length(mf.datasets)
  mf.nmodels <- length(mf.models)
  
  table_names <- c("Dataset", "Feats", "Classes", "Oracle", "Surrogate", 
                  "Accuracy", "Kappa", "AccuracyLower", "AccuracyUpper", "AccuracyNull", "AccuracyPValue", "McnemarPValue",
                  "MSE", "RMSE", "MAE", "COS", "AE", "APE", "Bias", "MeanLogLoss", "MAPE", "MASE", "MDAE",
                  "MSLE", "Percent_Bias", "RAE", "RMSLE", "RRSE", "RSE", "SMAPE", "SSE")
  
  table_data <- data.frame(matrix(ncol = length(table_names)))
  colnames(table_data) <- table_names
  
  mf.ids <- 1
  for(mf.ds in mf.datasets) {
    # mf.ds <- "badges2.csv"
    print(paste("Dataset (", mf.ids, "/", mf.nds,"): ", mf.ds), sep="")
    
    mf.datos <- load_dataset(paste(mf.ds_path, mf.ds, sep=""))
    # mf.datos$Class <- as.factor(sapply(mf.datos$Class, function(x) {
    #   if (x == '+') return('PLUS') else if (x == '-') return('MINUS') else return(make.names(x))}))
    mf.datos$Class <- correct_class_names(mf.datos$Class)
    mf.Train <- mf.datos
    mf.datos$Class <- as.factor(mf.datos$Class)
    classes <- unique(mf.datos$Class)
   
    # print(paste("Num. classes:", length(classes)))
    # print(paste("Num. feats:", ncol(datos)-1))
    # print(paste("Num. instances:", nrow(datos)))
    
    mf.i_om <- 0
    for (mf.om in mf.models) {
      # mf.om <- "NB"
      mf.i_om <- mf.i_om + 1
      print("---------------------")
      print(paste("Dataset:", mf.ds, "(", mf.ids, "/", mf.nds, "):", mf.ds))
      print(paste("Oracle Model", "(", mf.i_om, "/", mf.nmodels, "):", mf.om))
      print("---------------------")
      
      mf.ts_oracle <- list()
      
      # Oracle test labels
      load(sprintf("%stest_%s_%s.RData", mf.preds_path, mf.ds, mf.om))
      
      mf.ts_oracle$probs <- correct_missing_classes(preds$probs, classes)
      mf.ts_oracle$labs <- preds$crisp #probs_to_labels_crisp(mf.ds, "test", mf.om)
      
      # Surrogate test labels
      preds  <- surrogate_labels(train=NULL, test=NULL, 
                                      path_result=sprintf("%stest_%s_%s", mf.preds_path, mf.ds, mf.om), 
                                      models=mf.init$models)
      preds <- correct_missing_classes_surrogate(preds, classes, mf.models)
      
      mf.ts_surrogate <- separate_crisp_prob(preds, mf.models)
      
      mf.metrics <- list()
      mf.metrics$crisp <- metrics_crisp(mf.ts_oracle$labs, mf.ts_surrogate$labs, mf.init$models)
      mf.metrics$probs <- metrics_probs(mf.ts_oracle$probs, mf.ts_surrogate$probs, mf.init$models)
      
      # "Dataset", "Feats", "Classes", "Oracle", "Surrogate", 
      # "Accuracy", "Kappa", "AccuracyLower", "AccuracyUpper", "AccuracyNull", "AccuracyPValue", "McnemarPValue",
      # "MSE", "RMSE", "MAE", "COS"
      
      for (mf.sm in mf.models) {
        # mf.sm <- "c5.0"
        
        df <- data.frame(
          mf.ds, ncol(mf.datos)-1, length(classes), mf.om, mf.sm,
          mf.metrics$crisp[[mf.sm]]$Accuracy,
          mf.metrics$crisp[[mf.sm]]$Kappa,
          mf.metrics$crisp[[mf.sm]]$AccuracyLower,
          mf.metrics$crisp[[mf.sm]]$AccuracyUpper,
          mf.metrics$crisp[[mf.sm]]$AccuracyNull,
          mf.metrics$crisp[[mf.sm]]$AccuracyPValue,
          mf.metrics$crisp[[mf.sm]]$McnemarPValue,
          mf.metrics$probs[[mf.sm]]$MSE,
          mf.metrics$probs[[mf.sm]]$RMSE,
          mf.metrics$probs[[mf.sm]]$MAE,
          mf.metrics$probs[[mf.sm]]$COS,
          mf.metrics$probs[[mf.sm]]$AE, 
          mf.metrics$probs[[mf.sm]]$APE, 
          mf.metrics$probs[[mf.sm]]$Bias, 
          mf.metrics$probs[[mf.sm]]$MeanLogLoss, 
          mf.metrics$probs[[mf.sm]]$MAPE, 
          mf.metrics$probs[[mf.sm]]$MASE, 
          mf.metrics$probs[[mf.sm]]$MDAE,
          mf.metrics$probs[[mf.sm]]$MSLE, 
          mf.metrics$probs[[mf.sm]]$Percent_Bias, 
          mf.metrics$probs[[mf.sm]]$RAE, 
          mf.metrics$probs[[mf.sm]]$RMSLE, 
          mf.metrics$probs[[mf.sm]]$RRSE, 
          mf.metrics$probs[[mf.sm]]$RSE, 
          mf.metrics$probs[[mf.sm]]$SMAPE, 
          mf.metrics$probs[[mf.sm]]$SSE
        )
        colnames(df) <- table_names
        table_data <- rbind(table_data, df)
      }
    }
    mf.ids <- mf.ids + 1
  }
  table_data <- table_data[2:nrow(table_data),]
  table_data$Oracle <- as.factor(table_data$Oracle)
  save(table_data, file=paste(mf.path_base,"metafeats6.RData", sep=""))
  write.table(x=table_data, 
              file=paste(mf.analytics_path, "metafeats6.csv", sep=""),
              row.names = FALSE, sep=",")
  
  # load(file=paste(mf.path_base,"metafeats-kappa.RData", sep=""))

  # Maximum kappa
  
  # ground_truth <- metafeats[,"Class"]
  # max_kappa <- factor(apply(metafeats[,1:(ncol(metafeats)-2)], 1, function(x) {colnames(metafeats)[x == max(x)]}), levels <- mf.init$models)
  # metafeats[,12] == max_kappa
  # 
  # conf_mat <- caret::confusionMatrix(metafeats$Class, max_kappa, mode="everything")
  # conf_mat
  # conf_mat$overall["Accuracy"]
  # 
  # 
  # 
  # # METAMODEL
  # NF <- length(mf.datasets)
  # f_size <- length(mf.models)
  # 
  # fold_idx <- unlist(lapply(1:NF, function(x) {rep(x, f_size)}))
  # #fold_idx
  # 
  # meta_preds <- list()
  # # Llevar el dataset que vaja malament
  # dss <- metafeats[,"dataset"]
  # ground_truth <- metafeats[,"Class"]
  # metafeats <- metafeats[complete.cases(metafeats),1:length(mf.init$models)+1 ]
  # 
  # dss <- dss[complete.cases(metafeats)]
  # ground_truth <- dss[complete.cases(metafeats)]
  # for (fold in 1:NF) {
  #   print(paste("fold:", fold))
  #   ts <- metafeats[which(fold_idx == fold),1:ncol(metafeats)-1]
  #   tr <- metafeats[which(fold_idx != fold),]
  # 
  #   # ts <- ts[complete.cases(ts),]
  #   # tr <- tr[complete.cases(tr),]
  #   
  #   meta_preds[[fold]] <- probs_to_labels(learn_and_evaluate(tr, ts, "rf"))
  # 
  # }
  # 
  # meta_preds <- factor(unlist(meta_preds), levels=unique(metafeats$Class))
  # conf_mat <- caret::confusionMatrix(metafeats$Class, meta_preds, mode="everything")
  # conf_mat
  # conf_mat$overall["Accuracy"]
}

  # load(file=paste(path_base,"metafeats.RData", sep=""))
  # 
  # NF <- length(datasets)
  # f_size <- length(models)
  # 
  # fold_idx <- unlist(lapply(1:NF, function(x) {rep(x, f_size)}))
  # fold_idx
  # 
  # meta_preds <- list()
  # # Llevar el dataset que vaja malament
  # # metafeats <- metafeats[complete.cases(metafeats),1:length(init$models)] 
  # for (fold in 1:NF) {
  #   print(paste("fold:", fold))
  #   ts <- metafeats[which(fold_idx == fold),]
  #   tr <- metafeats[which(fold_idx != fold),]
  #   
  #   # ts <- ts[complete.cases(ts),]
  #   # tr <- tr[complete.cases(tr),]
  #   
  #   meta_preds[[fold]] <- probs_to_labels(learn_and_evaluate(tr, ts, "rf"))
  #   
  # }
  # meta_preds <- factor(unlist(meta_preds), levels=unique(metafeats$Class))
  # conf_mat <- caret::confusionMatrix(metafeats$Class, meta_preds, mode="everything")
  # conf_mat
  # conf_mat$overall["Accuracy"]
# }
