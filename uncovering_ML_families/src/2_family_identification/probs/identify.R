id.path_base <- "files/probs/"
id.ds_path <- "datasets/extension_caepia/"

maximum_similarity <- function(path_base, ds_path) {
  library(dplyr)
  source("src/2_family_identification/utils/utils.R")
  
  load(file=paste(id.path_base,"metafeats6.RData", sep=""))

  table_data_bin <- table_data[table_data$Classes==2,]
  table_data_mul <- table_data[table_data$Classes > 2,]
  
  #table_data <- table_data_mul
  
  id.path_base <- correct_path(id.path_base)
  id.ds_path <- correct_path(id.ds_path)
  id.preds_path <- correct_path(paste(id.path_base, "predictions", sep=""))
  id.analytics_path <- correct_path(paste(id.path_base, "analytics", sep=""))
  
  id.init <- init_probs(id.path_base, id.ds_path)
  
  id.datasets <- id.init$datasets#[2:length(id.init$datasets)]
  
  id.models <- id.init$models
  
  id.nds = length(id.datasets)
  id.nmodels <- length(id.models)
  
  metric_names <- c("Accuracy", "Kappa", "AccuracyLower", "AccuracyUpper", "AccuracyNull", "AccuracyPValue",
                   "MSE", "RMSE", "MAE", "COS", "AE", "APE", "Bias", "MeanLogLoss", "MAPE", "MASE",
                   "MSLE", "RAE", "RMSLE", "RSE", "SMAPE", "SSE")
  # metric_names <- c("Kappa", "AE", "MAE")
  min_metrics <- c("MSE", "RMSE", "MAE", "AE", "APE", "Bias", "MeanLogLoss", 
                   "MAPE", "MASE", "MDAE", "MSLE", "Percent_Bias", "RAE", "RMSLE", "RSE", "SMAPE", 
                   "SSE")
  max_metrics <- c("Accuracy", "Kappa", "AccuracyLower", "AccuracyUpper", "AccuracyNull", 
                   "AccuracyPValue", "COS")
  
  id.datasets <- unique(table_data$Dataset)
  
  id.ids <- 1
  most_similar <- list()
  ranking_table <- list()
  for (id.ds in id.datasets) {
    # id.ds <- "badges2.csv"
    most_similar[[id.ds]] <- list()
    id.ds_data <- table_data[table_data[,"Dataset"] == id.ds,]
    
    for (id.om in id.models) {
      # id.om <- "NB"
      most_similar[[id.ds]][[id.om]] <- list()
      id.oracle_data <- id.ds_data[id.ds_data[,"Oracle"] == id.om,]
      
      for (id.m in metric_names) {
         # id.m <- "MAPE"
        if (id.m %in% max_metrics) {
          id.pos <- which(id.oracle_data[[id.m]] == max(id.oracle_data[[id.m]]))
        } else {
          id.pos <- which(sapply(as.numeric(id.oracle_data[[id.m]]), abs) == 
                            min(sapply(as.numeric(id.oracle_data[[id.m]]), abs)))
        }
        if(length(id.pos) == 0) {
          id.pos <- sample(11, 1)
        }
        if (length(id.pos) > 1) {
          set.seed(3)
          id.pos <- sample(id.pos, 1)
        }
        
        # most_similar[[id.ds]][[id.om]][[id.m]] <- tryCatch(id.oracle_data[[id.pos, "Surrogate"]], 
        #          error = function(e) {return(NaN)})
        most_similar[[id.ds]][[id.om]][[id.m]] <- id.oracle_data[[id.pos, "Surrogate"]]
                                                  
      }
    }
  }
  
  real_labels <- list()
  pred_labels <- list()
  
  i_oracle <- 0
  for (id.ds in id.datasets) {
    for (id.om in id.models) {  
      i_oracle <- i_oracle + 1
      pred_labels[[i_oracle]] <- list()
      real_labels[[i_oracle]] <- id.om
      for (id.m in metric_names) {  
        pred_labels[[i_oracle]][[id.m]] <- most_similar[[id.ds]][[id.om]][[id.m]]
      }
    }
  }
  
  real_labels <- factor(as.vector(real_labels), levels=id.models)
  metric_labels <- list()
  
  for (id.m in metric_names) {
    metric_labels[[id.m]] <- c()
    for (i_oracle in 1:length(real_labels)) {
      metric_labels[[id.m]] <- c(metric_labels[[id.m]], pred_labels[[i_oracle]][[id.m]])
    }
    metric_labels[[id.m]] <- factor(as.vector(metric_labels[[id.m]]), levels=id.models)
  }
  conf_mats <- list()
  for (id.m in metric_names) {
    metric_preds <- metric_labels[[id.m]]
    # reference <- real_labels[complete.cases(metric_preds)]
    reference <- real_labels#[metric_preds]
    # data <- metric_preds[complete.cases(metric_preds)]
    data <- metric_preds#[metric_preds]
    conf_mat <- caret::confusionMatrix(reference = reference, data = data, mode="everything")
    print(paste(id.m, "Accuracy:", conf_mat$overall["Accuracy"]))
    conf_mats[[id.m]] <- conf_mat
    
    write.table(x=conf_mats[[id.m]]$table, 
                file=paste(id.analytics_path, sprintf("max_sim_mul_conf_mat_%s.csv", id.m), sep=""),
                row.names = TRUE, sep=",")
    
    capture.output(
    cat(paste("\n\n---------------", id.m, "---------------\n")),
    # cat(paste("COMPLETE EXAMPLES: ",
    #           length(metric_labels[[id.m]][complete.cases(metric_labels[[id.m]])]),
    #           "/", length(id.models)*length(id.datasets),"\n",
    #     sep="")),
    cat(paste("COMPLETE EXAMPLES: ",
              length(metric_labels[[id.m]]),
              "/", length(id.models)*length(id.datasets),"\n",
              sep="")),
    print(conf_mats[[id.m]]), 
    file=paste(id.path_base, sprintf("analytics/max_sim_mul_conf_mat_%s.txt", id.m), sep=""))
    

  }
  
  # # Save confusion matrices
  # capture.output(
  #   for (id.m in metric_names) {
  #     cat(paste("\n\n---------------", id.m, "---------------\n"));
  #     cat(paste("COMPLETE EXAMPLES: ", 
  #         length(metric_labels[[id.m]][complete.cases(metric_labels[[id.m]])]), "/", length(id.models)*length(id.datasets), "\n", sep=""))
  #     print(conf_mats[[id.m]])}, 
  # file=paste(id.path_base, "analytics/conf_mats.txt", sep=""))
}

id.path_base <- "files/probs/"
id.ds_path <- "datasets/extension_caepia/"
feats <- c("Kappa", "MAE", "Kappa_MAE")
metamodel <- function(path_base, ds_path, feats) {
  library(dplyr)
  library(randomForest)
  source("src/2_family_identification/utils/utils.R")
  
  load(file=paste(id.path_base,"metafeats6.RData", sep=""))
  
  #table_data_bin <- table_data[table_data$Classes==2,]
  #table_data_mul <- table_data[table_data$Classes > 2,]
  
  # table_data <- table_data_mul
  
  id.path_base <- correct_path(id.path_base)
  id.ds_path <- correct_path(id.ds_path)
  id.preds_path <- correct_path(paste(id.path_base, "predictions", sep=""))
  id.analytics_path <- correct_path(paste(id.path_base, "analytics", sep=""))
  
  id.init <- init_probs(id.path_base, id.ds_path)
  
  id.datasets <- id.init$datasets#[2:length(id.init$datasets)]
  id.datasets <- unique(table_data$Dataset)
  
  
  id.models <- id.init$models
  
  id.nds = length(id.datasets)
  id.nmodels <- length(id.models)

  meta_datasets <- build_metadatasets(table_data, feats, id.datasets, id.models)
  identify_metamodel(meta_datasets, id.datasets, id.models, table_data, id.analytics_path)
}
