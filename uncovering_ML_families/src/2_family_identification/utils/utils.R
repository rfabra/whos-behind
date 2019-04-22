init_env <- function(base_dir, path_datasets) {
  source("src/common/common.R")
  source("src/common/queries.R")
  source("src/2_family_identification/utils/models.R")
  
  path_datasets <- correct_path(path_datasets)
  # Load datasets
  datasets <- list.files(path_datasets, pattern=".csv")
  datasets <- datasets[!grepl("coltypes", datasets)]
  
  dir.create(file.path(base_dir), showWarnings = FALSE)
  dir.create(file.path(base_dir, "analytics"), showWarnings = FALSE)
  dir.create(file.path(base_dir, "locks"), showWarnings = FALSE)
  dir.create(file.path(base_dir, "predictions"), showWarnings = FALSE)
  dir.create(file.path(base_dir, "surrogate_datasets"), showWarnings = FALSE)
  
  return(list("datasets" = datasets, 
       "models" = models)
  )
}

init_crisp <- function(base_dir, path_datasets) {
  source("src/common/methods.R")
  return(init_env(base_dir, path_datasets))
  
}

init_probs <- function(base_dir, path_datasets) {
  source("src/common/methods_prob.R")
  return(init_env(base_dir, path_datasets))
}

learn_and_evaluate <- function(train, test, method, path_result="") {
  preds <- list()
  set.seed(3)
  if (path_result == "") {
    preds <- runmethods(method, train, test)
  } else {
    if (!file.exists(path_result)) {
        print(paste("Learn and evaluate:", path_result))
        preds <- runmethods(method, train, test)
        print(predictionsState(preds$crisp, length(unique(train$Class))))
        save(preds, file=path_result)
    } else {
      print(paste("Already processed:", path_result))
      load(path_result)
    }
  }
  return(preds)
}

surrogate_labels <- function(train, test, models, path_result="") {
  preds <- list()
  if (path_result == "") {
    for (sm in models) {
      preds[[sm]] <- learn_and_evaluate(train, test, sm)
    }
  } else {
    for (sm in models) {
      preds[[sm]] <- learn_and_evaluate(train, test, sm,
                                        sprintf("%s_%s.RData", path_result, sm))
    }
  }
  return(preds)
}

dataset_entropy <- function(predictions, labels) {
  predictions <- as.data.frame(predictions)
  lab_freqs <- list()
  for (lab in labels) {
    lab_freqs[[sprintf("class_%s", lab)]] <- data.frame("kk" = apply(predictions, 1, function(x) {sum(x==lab)}))
  }
  
  entropy_df <- data.frame("kk" = apply(as.data.frame(lab_freqs), 1, entropy))
  colnames(entropy_df) <- sprintf("label_entropy", lab)
  entropy_df
}

split_dataset_entropy <- function(label_entropies, criterion="train-boundaries") {
  informative_examples <- label_entropies > 0

  if(criterion == "train-boundaries") {
    train_idx <- informative_examples
  }
  else if (criterion == "test-boundaries") {
    train_idx <- !informative_examples
  }
  
  train_idx
}

metafeat_kappa <- function(ground_truth, predictions, models) {
  #ground_truth <- ts_oracle_labs
  #predictions <- ts_surrogate_labs
  i_sm <- 0
  kappas <- list()
  for (sm in models) {
    #sm <- "rda"
    print(sprintf("Kappa. Surrogate %s", sm))
    i_sm <- i_sm + 1
    predictions[[sm]] <- factor(predictions[[sm]], levels=levels(ground_truth))
    # print(paste("Reference factors:", levels(ground_truth)))
    # print(paste("Data factors:", levels(predictions[[sm]])))
    conf_mat <- caret::confusionMatrix(reference = ground_truth, data = predictions[[sm]], mode="everything")
    kappas[[sm]] <- conf_mat$overall["Kappa"]
  }
  kappas
}

metafeat_mse <- function(probs_oracle, probs_surrogate, models) {
  library(Metrics)
  probs_oracle <- ts_oracle_probs
  probs_surrogate <- ts_surrogate_probs
  i_sm <- 0
  mse_feats <- list()
  for (sm in models) {
    i_sm <- i_sm + 1
    mse_feats[[sm]] <- mean(sapply(1:nrow(probs_oracle), function(x) {
      mse(as.numeric(probs_oracle[x,]), as.numeric(probs_surrogate[["c5.0"]][x,]))}))
  }
  mse_feats
}

correct_class_names <- function(original_names) {
  return(
    as.factor(sapply(original_names, function(x) {
    if (x == '+') return('PLUS') 
    else if (x == '-') return('MINUS') 
    else return(make.names(x))
      }))
  )
}

correct_missing_classes <- function(probs, Classes) {
  # probs <- predictions[[sm]]$probs
  for (c in Classes) {
    if (!c %in% colnames(probs)) {
      probs <- cbind(probs, data.frame(x=rep(0.0, nrow(probs))))
      colnames(probs)[ncol(probs)] <- c
    }
  }
  return(probs[,sort(as.vector(Classes))])
}

correct_missing_classes_surrogate <- function(predictions, Classes, models) {
  # predictions <- preds
  # Classes <- classes
  # models <- mf.models
  
  # out <- list()
  
  for (sm in models) {
    predictions[[sm]]$probs <- correct_missing_classes(predictions[[sm]]$probs, Classes)
  }
  return(predictions)
}


class_probabilities <- function(probs_oracle, probs_surrogate, models) {
  library(Metrics)
  
  # probs_oracle <- mf.ts_oracle_probs
  # probs_surrogate <- mf.ts_surrogate_probs

  classes <- colnames(probs_oracle)
  
  #probs_oracle <- correct_missing_classes(probs_oracle, classes)
  for (sm in models) {
    probs_surrogate[[sm]] <- correct_missing_classes(probs_surrogate[[sm]], classes)
  }
  
  class_probs <- list()
  # colnames
  i_sm <- 0
  for (c in classes) {
    class_probs[[c]] <- list()
    class_probs[[c]][["oracle"]] <- probs_oracle[,c]
    for (sm in models) {
    i_sm <- i_sm + 1
      class_probs[[c]][[sm]] <- probs_surrogate[[sm]][,c]
    }
  }
  #class_probs <- as.data.frame(class_probs)
  return(class_probs)
}

class_mse <- function(c_probs, models) {
  library(Metrics)
  
  #c_probs <- mf.class_probs
  
  classes <- names(c_probs)
  class_mse <- list()
  # colnames
  i_sm <- 0
  for (c in classes) {
    class_mse[[c]] <- list()
    #class_probs[[c]][["oracle"]] <- probs_oracle[,c]
    for (sm in models) {
      i_sm <- i_sm + 1
      class_mse[[c]][[sm]] <- mse(c_probs[[c]]$oracle, c_probs[[c]][[sm]])
    }
  }
  #class_probs <- as.data.frame(class_probs)
  return(class_mse)
}

class_rmse <- function(c_probs, models) {
  library(Metrics)
  
  #c_probs <- mf.class_probs
  
  classes <- names(c_probs)
  class_rmse <- list()
  # colnames
  i_sm <- 0
  for (c in classes) {
    class_rmse[[c]] <- list()
    #class_probs[[c]][["oracle"]] <- probs_oracle[,c]
    for (sm in models) {
      i_sm <- i_sm + 1
      class_rmse[[c]][[sm]] <- rmse(c_probs[[c]]$oracle, c_probs[[c]][[sm]])
    }
  }
  #class_probs <- as.data.frame(class_probs)
  return(class_rmse)
}

class_cosine <- function(c_probs, models) {
  library(philentropy)
  
  #c_probs <- mf.class_probs
  
  classes <- names(c_probs)
  class_cosine <- list()
  # colnames
  i_sm <- 0
  for (c in classes) {
    class_cosine[[c]] <- list()
    #class_probs[[c]][["oracle"]] <- probs_oracle[,c]
    for (sm in models) {
      i_sm <- i_sm + 1
      class_cosine[[c]][[sm]] <- distance(rbind(c_probs[[c]]$oracle, c_probs[[c]][[sm]]), method = "cosine")
    }
  }
  #class_probs <- as.data.frame(class_probs)
  return(class_cosine)
}

class_mae <- function(c_probs, models) {
  library(Metrics)
  
  c_probs <- mf.class_probs
  
  classes <- names(c_probs)
  class_mae <- list()
  # colnames
  i_sm <- 0
  for (c in classes) {
    class_mae[[c]] <- list()
    #class_probs[[c]][["oracle"]] <- probs_oracle[,c]
    for (sm in models) {
      i_sm <- i_sm + 1
      class_mae[[c]][[sm]] <- mae(c_probs[[c]]$oracle, c_probs[[c]][[sm]])
    }
  }
  #class_probs <- as.data.frame(class_probs)
  return(class_mae)
}

correct_path <- function(path) {
  if (substr(path,start = nchar(path), stop=nchar(path)) != "/") {
    path <- paste(path, "/", sep="")
  }
  return(path)
}

metafeat_class_props <- function(oracle_labels, labels) {
  result <- c()
  #colnames(result) <- labels
  for (l in labels) {
    result <- c(result, sum(oracle_labels == l))#/length(oracle_labels)
  } 
  
  #colnames(result) <- sprintf("prop.c.%s",labels) 
  return(result)
}

probs_to_labels <- function(prob_list, class_levels) {
  labels <- as.vector(apply(prob_list,1,function (x) {colnames(prob_list)[which.max(x)]}))
  return(factor(labels, levels=unique(labels)))
}

probs_to_labels_crisp <- function(crisp_dir, dset, train_test, omodel, smodel="") {

  crisp_dir <- correct_path(crisp_dir)
  path <- ""
  if (smodel != "") {
    path <- sprintf("%spredictions/%s_%s_%s_%s.RData", 
                    crisp_dir, train_test, dset, omodel, smodel)
  } else {  
    path <- sprintf("%spredictions/%s_%s_%s.RData", 
                    crisp_dir, train_test, dset, omodel)
  }
  print(paste("Loading crisp prediction: ", path))
  load(path)
  preds <- correct_class_names(preds)
  return(factor(preds, levels=unique(preds)))
}

surrogate_labels_crisp <- function(dset, omodel, smodels) {
  # print("Inside surrogate labels crisp")
  print(dset)
  print(omodel)
  print(smodels)
  surrogate_preds <- list()
  for (sm in smodels) {
    surrogate_preds[[sm]] <- probs_to_labels_crisp(dset, "test", omodel, sm)
  }
  return(surrogate_preds)
}

separate_crisp_prob <- function(predictions, mf.models) {
  out <- list()
  out$labs <- list()
  out$probs <- list()
  for (sm in mf.models) {
    out$probs[[sm]] <- predictions[[sm]]$probs
    out$labs[[sm]] <- predictions[[sm]]$crisp
  }
  return(out)
}

metrics_probs <- function(o_probs, s_probs, models) {
  # o_probs <- mf.ts_oracle$probs
  # s_probs <- mf.ts_surrogate$probs
  # models <- mf.models
  
  library(Metrics)
  library(lsa)
  
  metrics <- list()
  for (sm in models) {
    # sm <- "rda"
    metrics[[sm]] <- list()
    metrics[[sm]][["MSE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(mse(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["RMSE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(rmse(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["MAE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(mae(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["COS"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(lsa::cosine(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["AE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(ae(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["APE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(ape(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["Bias"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(bias(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["MeanLogLoss"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(logLoss(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
      
    metrics[[sm]][["MAPE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(mape(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["MASE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(mase(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
      #d<- distance(o_probs[1,], s_probs[[sm]][1,], method = "cosine")
    
    metrics[[sm]][["MDAE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(mdae(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
  
    metrics[[sm]][["MSLE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(msle(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["Percent_Bias"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(percent_bias(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["RAE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(rae(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["RMSLE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(rmsle(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["RRSE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(rrse(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))  
    
    metrics[[sm]][["RSE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(rse(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["SMAPE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(smape(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
    
    metrics[[sm]][["SSE"]] <- mean(sapply(1:nrow(o_probs), function(x) {
      return(sse(as.numeric(o_probs[x,]), as.numeric(s_probs[[sm]][x,])))}))
  }
 
  return(metrics)
}

metrics_crisp <- function(ground_truth, predictions, models) {
  # ground_truth <- mf.ts_oracle$labs
  # predictions <- mf.ts_surrogate$labs
  metrics <- list()
  for (sm in models) {
    #sm <- "rda"
    metrics[[sm]] <- list()
    predictions[[sm]] <- factor(predictions[[sm]], levels=levels(ground_truth))
    conf_mat <- caret::confusionMatrix(data = predictions[[sm]], reference = ground_truth, mode="everything")

    metrics[[sm]]$Accuracy <- conf_mat$overall["Accuracy"]
    metrics[[sm]]$Kappa <- conf_mat$overall["Kappa"]
    metrics[[sm]]$AccuracyLower <- conf_mat$overall["AccuracyLower"]
    metrics[[sm]]$AccuracyUpper <- conf_mat$overall["AccuracyUpper"]
    metrics[[sm]]$AccuracyNull <- conf_mat$overall["AccuracyNull"]
    metrics[[sm]]$AccuracyPValue <- conf_mat$overall["AccuracyPValue"]
    metrics[[sm]]$McnemarPValue <- conf_mat$overall["McnemarPValue"]
  }
  
  return(metrics)
}

compute_ranking <- function(metric_data) {
  library(dplyr)
  #ordered <- order(metric_data)[1:n_top]
  
  for (o in ordered) {
    
  }
  return( which(metric_data >= ordered[n_top]) )
}

build_metadatasets <- function(full_data, feats, datasets, models) {
  
  # full_data <- table_data
  # datasets <- id.datasets
  # models <- id.models
  meta_datasets <- list()
  ds_names <- c()
  ids <- 1
  for (ds in datasets) {
    # ds <- "badges2.csv"
    ds_data <- full_data[full_data[,"Dataset"] == ds,]
    
    i_om <- 1
    for (om in models) {
      # om <- "NB"
      oracle_data <- ds_data[ds_data[,"Oracle"] == om,]
      # ds_names <- c(ds_names, ds)
      if ("Kappa" %in% feats) {
        if (ids == 1 && i_om == 1) {
          meta_datasets$Kappa <- data.frame(t(oracle_data$Kappa), om)
          colnames(meta_datasets$Kappa) <- c(paste("Kappa", oracle_data$Surrogate, sep="."), "Class")
        } else {
          df <- data.frame(t(oracle_data$Kappa), om)
          colnames(df) <- c(paste("Kappa", oracle_data$Surrogate, sep="."), "Class")
          meta_datasets$Kappa <- rbind(meta_datasets$Kappa, df)
        }
      } 
      if ("MAE" %in% feats) {
        if (ids == 1 && i_om == 1) {
          meta_datasets$MAE <- data.frame(t(oracle_data$MAE), om)
          colnames(meta_datasets$MAE) <- c(paste("MAE", oracle_data$Surrogate, sep="."), "Class")
        } else {
          df <- data.frame(t(oracle_data$MAE), om)
          colnames(df) <- c(paste("MAE", oracle_data$Surrogate, sep="."), "Class")
          meta_datasets$MAE <- rbind(meta_datasets$MAE, df)        
        }
      }
      if ("Kappa_MAE" %in% feats) {
        if (ids == 1 && i_om == 1) {
          meta_datasets$Kappa_MAE <- data.frame(t(oracle_data$Kappa), t(oracle_data$MAE), om)
          colnames(meta_datasets$Kappa_MAE) <- c(paste("Kappa", oracle_data$Surrogate, sep="."), 
                                                 paste("MAE", oracle_data$Surrogate, sep="."), 
                                                 "Class")
        } else {
          df <- data.frame(t(oracle_data$Kappa), t(oracle_data$MAE), om)
          colnames(df) <- c(paste("Kappa", oracle_data$Surrogate, sep="."), 
                            paste("MAE", oracle_data$Surrogate, sep="."), 
                            "Class")
          meta_datasets$Kappa_MAE <- rbind(meta_datasets$Kappa_MAE, df)        
        }
      }
      i_om <- i_om + 1
    }
    ids <- ids + 1
  }
  return(meta_datasets)
  #return(list("meta_datasets"=meta_datasets, "ds_names"=ds_names))
}

save_confmat <- function(predicted, ground_truth, out_path) {
  # predicted <- meta_preds2
  # ground_truth <- meta_datasets[[f]]$Class
  # meta_preds2 <- factor(unlist(meta_preds[[f]]), levels=unique(meta_datasets[[f]]$Class))
  conf_mat <- caret::confusionMatrix(data = predicted, reference = ground_truth, mode="everything")
  conf_mat
  
  print(conf_mat$overall["Accuracy"])
  write.table(x=conf_mat$table, 
              file=paste(dest_path, sprintf("metamodel_all_conf_mat_%s_rf.csv", f), sep=""),
              row.names = TRUE, sep=",")
  capture.output(
    print(conf_mat), 
    file=out_path, sep="")
}

identify_metamodel <- function(meta_datasets, datasets, models, table_data, dest_path) {
  # datasets <- id.datasets
  # models <- id.models
  # full_data <- table_data
  # dest_path <- "/tmp/"
  print(dest_path)
  dest_path <- correct_path(dest_path)
  print(dest_path)
  NF <- length(datasets)
  f_size <- length(models)
  
  fold_idx <- unlist(lapply(1:NF, function(x) {rep(x, f_size)}))
  fold_ds <- unlist(lapply(datasets, function(x) {return(rep(x, f_size))}))

  #fold_idx
  meta_preds <- list()
  bin_preds <- list()
  mul_preds <- list()
  bin_true <- list()
  mul_true <- list()
  for (f in names(meta_datasets)) {
    # f <- "Kappa"
    print(f)
    meta_preds[[f]] <- list()
    bin_preds[[f]] <- list()
    mul_preds[[f]] <- list()
    bin_true[[f]] <- list()
    mul_true[[f]] <- list()
    # Llevar el dataset que vaja malament
    # metafeats <- metafeats[complete.cases(metafeats),1:length(init$models)] 
    for (fold in 1:NF) {
      # fold <- 1
      print(paste("fold:", fold))
      ts <- meta_datasets[[f]][which(fold_idx == fold),]
      ts <- ts[,!colnames(ts) %in% "Class"]
      tr <- meta_datasets[[f]][which(fold_idx != fold),]
      
      set.seed(34)

      model <- caret::train(Class ~ ., data = tr, preProc = c("center", "scale"),
                            method = "rf", 
                            trControl = trainControl(method="none")) # learn_and_evaluate(tr, ts, "rf")
      # model <- caret::train(Class ~ ., data = tr, preProc = c("center", "scale"), 
      #                       method = "knn",  tuneGrid=data.frame(k = 3),
      #                       trControl = trainControl(method="none")) # learn_and_evaluate(tr, ts, "rf")
      meta_preds[[f]][[fold]] <- predict(model, newdata = ts, type="raw")
      # meta_preds[[fold]] <- learn_and_evaluate(tr, ts, "rf")
      #meta_preds[[fold]] <- preds$crisp#probs_to_labels(learn_and_evaluate(tr, ts, "rf"))
      
      nClasses <- unique(table_data[table_data$Dataset == datasets[fold],"Classes"])
      if (nClasses == 2) {
        bin_preds[[f]][[fold]] <- meta_preds[[f]][[fold]]
        bin_true[[f]][[fold]] <- meta_datasets[[f]][which(fold_idx == fold),]$Class
      }
      else{
        mul_preds[[f]][[fold]] <- meta_preds[[f]][[fold]]
        mul_true[[f]][[fold]] <- meta_datasets[[f]][which(fold_idx == fold),]$Class
      }
    }
    # ALL predictions
    # print("ALL predictions")
    # predictions <- factor(unlist(meta_preds[[f]]), levels=unique(meta_datasets[[f]]$Class))
    # ground <- meta_datasets[[f]]$Class
    # dest_path <- "files/probs/analytics/"
    # save_confmat(predicted = predictions, ground_truth = ground, 
    #              out_path = paste(dest_path, sprintf("metamodel_all_conf_mat_%s_rf.csv", f), sep=""))
    # 
    
    meta_preds2 <- factor(unlist(meta_preds[[f]]), levels=unique(meta_datasets[[f]]$Class))
    conf_mat <- caret::confusionMatrix(meta_datasets[[f]]$Class, meta_preds2, mode="everything")
    conf_mat
    print(f)
    print(conf_mat$overall["Accuracy"])
    write.table(x=conf_mat$table,
                file=paste(dest_path, sprintf("metamodel_all_conf_mat_%s_rf.csv", f), sep=""),
                row.names = TRUE, sep=",")
    capture.output(
      print(conf_mat),
      file=paste(dest_path, sprintf("metamodel_all_conf_mat_%s_rf.txt", f), sep=""))
    
    # Binary predictions
    print("BINARY predictions")
    bin_preds[[f]] <- bin_preds[[f]][!sapply(bin_preds[[f]], is.null)]
    bin_true[[f]] <- bin_true[[f]][!sapply(bin_true[[f]], is.null)]
    meta_preds2 <- factor(unlist(bin_preds[[f]]), levels=unique(meta_datasets[[f]]$Class))
    bin_true2 <- factor(unlist(bin_true[[f]]), levels=unique(meta_datasets[[f]]$Class))
    conf_mat <- caret::confusionMatrix(bin_true2, meta_preds2, mode="everything")
    conf_mat
    print(f)
    print(conf_mat$overall["Accuracy"])
    
    write.table(x=conf_mat$table, 
                file=paste(dest_path, sprintf("metamodel_bin_conf_mat_%s_rf.csv", f), sep=""),
                row.names = TRUE, sep=",")
    capture.output(
      print(conf_mat), 
      file=paste(dest_path, sprintf("metamodel_bin_conf_mat_%s_rf.txt", f), sep=""))
    
    # Multiclass predictions
    print("MULTICLASS predictions")
    mul_preds[[f]] <- mul_preds[[f]][!sapply(mul_preds[[f]], is.null)]
    mul_true[[f]] <- mul_true[[f]][!sapply(mul_true[[f]], is.null)]
    mul_true2 <- factor(unlist(mul_true[[f]]), levels=unique(meta_datasets[[f]]$Class))
    meta_preds2 <- factor(unlist(mul_preds[[f]]), levels=unique(meta_datasets[[f]]$Class))
    conf_mat <- caret::confusionMatrix(mul_true2, meta_preds2, mode="everything")
    conf_mat
    print(f)
    print(conf_mat$overall["Accuracy"])
    write.table(x=conf_mat$table, 
                file=paste(dest_path, sprintf("metamodel_mul_conf_mat_%s_rf.csv", f), sep=""),
                row.names = TRUE, sep=",")
    capture.output(
      print(conf_mat), 
      file=paste(dest_path, sprintf("metamodel_mul_conf_mat_%s_rf.txt", f), sep=""))
  
  }
}