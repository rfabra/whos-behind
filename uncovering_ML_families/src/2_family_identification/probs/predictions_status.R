# options( java.parameters = "-Xmx6g" )
# options(warn=1)
# 
ps.path_base <- "files/probs/"
ps.ds_path <- "datasets/extension_caepia/"

try_load_preds <- function(path) {
  error <- tryCatch(load(path), error = function(e) {print(e); return(TRUE)})
  if (is.logical(error)) {
    preds <- list()  
  }
  return(preds)
}

prediction_status <- function(ps.path_base, ps.ds_path) {
  # ps.path_base <- "2_family_identification/files/train_test_prob/"
  # ps.ds_path <- "datasets/extension_caepia/"

  source("src/2_family_identification/utils/utils.R")
  ps.path_base <- correct_path(ps.path_base)
  ps.ds_path <- correct_path(ps.ds_path)
  ps.preds_path <- correct_path(paste(ps.path_base, "predictions", sep=""))
  ps.analytics_path <- correct_path(paste(ps.path_base, "analytics", sep=""))
  
  ps.init <- init_probs(ps.path_base, ps.ds_path)
  
  ps.datasets <- ps.init$datasets#[1:30]
  # ps.datasets <- ps.datasets[ps.datasets != "colic.csv"]
  # ps.datasets <- ps.datasets[ps.datasets != "hepatitis.csv"]
  # ps.datasets <- ps.datasets[ps.datasets != "lymph.csv"]
  # ps.datasets <- ps.datasets[ps.datasets != "lung-cancer.csv"]
  # ps.datasets <- ps.datasets[ps.datasets != "chess-KRVKP.csv"]
  # ps.datasets <- ps.datasets[ps.datasets != "breast-cancer.csv"]
  # ps.datasets <- ps.datasets[ps.datasets != "eucalyptus.csv"]
  # ps.datasets <- ps.datasets[ps.datasets != "mushrooms.csv"]
  # ps.datasets <- ps.datasets[ps.datasets != "musk1.csv"]
  ps.models <- ps.init$models
  
  # datasets <- datasets[1:10]
  # models <- c("c5.0", "mlp_3", "svmRadialCost_C2_5")
  
  ps.nds = length(ps.datasets)
  ps.nmodels <- length(ps.models)
  
  ps.col_names <- c("id", "Dataset",  "Classes", "Features", "Instances", "Oracle", "Oracle.output")
  ps.csv_control <- data.frame(matrix(ncol=length(ps.col_names) + length(ps.models)))
  ps.col_names <- c(ps.col_names, ps.models)
  colnames(ps.csv_control) <- ps.col_names
  
  ps.ids <- 0
  for(ps.ds in ps.datasets) {
    ps.ids <- ps.ids + 1
    #ds <- "diabetes.csv"
  
    print(paste("Dataset (", ps.ids, "/", ps.nds,"): ", ps.ds), sep="")
    
    ps.datos <- load_dataset(paste(ps.ds_path, ps.ds, sep=""))
    ps.datos$Class <- as.factor(ps.datos$Class)
    ps.classes <- unique(ps.datos$Class)
    
    print(paste("Num. classes:", length(ps.classes)))
    print(paste("Num. feats:", ncol(ps.datos)-1))
    print(paste("Num. instances:", nrow(ps.datos)))
    
    ps.i_om <- 0
    for (ps.om in ps.models) {
      #om <- "c5.0"
      ps.i_om <- ps.i_om + 1
      print("---------------------")
      print(paste("Dataset:", ps.ds, "(", ps.ids, "/", ps.nds, "):", ps.ds))
      print(paste("Oracle Model", "(", ps.i_om, "/", ps.nmodels, "):", ps.om))
      print("---------------------")
      
      # Surrogate training labels
      preds <- try_load_preds(sprintf("%strain_%s_%s.RData", ps.preds_path, ps.ds, ps.om))
      if (length(preds) == 0) next
      
      ps.tr_oracle <- list()
      ps.tr_oracle$labs <- preds$crisp
      ps.tr_oracle$probs <- preds$probs
      
      # Surrogate test labels
      preds <- try_load_preds(sprintf("%stest_%s_%s.RData", ps.preds_path, ps.ds, ps.om))
      if (length(preds) == 0) next
      ps.ts_oracle <- list()
      
      ps.ts_oracle$labs <- preds$crisp
      ps.ts_oracle$probs <- preds$crisp
      
      ps.ps <- predictionsStateProb(ps.tr_oracle$probs)
      ps.pred_state <- ps.ps
      if (ps.ps == "OK") {
        ps.pred_state <- predictionsState(ps.tr_oracle$labs, length(ps.classes)) 
      }
      
      ps.df <- data.frame("id"=ps.ids,
                       "Dataset"=ps.ds,
                       "Classes"=length(ps.classes),
                       "Features"=ncol(ps.datos)-1,
                       "Instances"=nrow(ps.datos),
                       "Oracle"=ps.om,
                       "Oracle.output"=ps.pred_state)
      
      for (ps.sm in ps.models) {
        #sm <- "c5.0"
        preds <- try_load_preds(sprintf("%stest_%s_%s_%s.RData", ps.preds_path, ps.ds, ps.om, ps.sm))
        # if (length(preds) == 0) next
        
        ps.ps <- predictionsStateProb(preds$probs)
        ps.pred_state <- ps.ps
        if (ps.ps == "OK") {
          ps.pred_state <- predictionsState(preds$crisp, length(ps.classes)) 
        }
        
        ps.df <- cbind(ps.df, data.frame(sm=ps.pred_state))
        colnames(ps.df)[ncol(ps.df)] <- ps.sm
      }
      ps.csv_control <- rbind(ps.csv_control, ps.df)
    }
  }
  
  write.table(x=ps.csv_control[2:nrow(ps.csv_control),], 
              file=paste(ps.analytics_path, "predictions_status.csv", sep=""), 
              row.names = FALSE, sep=",")
}
