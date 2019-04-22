options( java.parameters = "-Xmx6g" )
options(warn=1)

prediction_status <- function(path_base) {
  # path_base <- "2_family_identification/files/surrogate_output/"
  source("2_family_identification/utils.R")
  path_base <- correct_path(path_base)
  preds_path <- correct_path(paste(path_base, "predictions", sep=""))
  analytics_path <- correct_path(paste(path_base, "analytics", sep=""))
  ds_path <- correct_path("datasets/clean")
  
  init <- init_train_test(ds_path)
  
  datasets <- init$datasets
  models <- init$models
  
  nds = length(datasets)
  nmodels <- length(models)
  
  col_names <- c("id", "Dataset",  "Classes", "Features", "Instances", "Oracle", "Oracle.output")
  csv_control <- data.frame(matrix(ncol=length(col_names)))
  #col_names <- c(col_names, models)
  colnames(csv_control) <- col_names
  
  ids <- 1
  
  for(ds in datasets) {
    #ds <- "diabetes.csv"
  
    print(paste("Dataset (", ids, "/", nds,"): ", ds), sep="")
    
    datos <- load_dataset(paste(ds_path, ds, sep=""))
    datos$Class <- as.factor(datos$Class)
    classes <- unique(datos$Class)
    
    print(paste("Num. classes:", length(classes)))
    print(paste("Num. feats:", ncol(datos)-1))
    print(paste("Num. instances:", nrow(datos)))
    
    i_om <- 0
    for (om in models) {
      #om <- "c5.0"
      i_om <- i_om + 1
      print("---------------------")
      print(paste("Dataset:", ds, "(", ids, "/", nds, "):", ds))
      print(paste("Oracle Model", "(", i_om, "/", nmodels, "):", om))
      print("---------------------")
      
      # Surrogate training labels
      load(sprintf("%strain_%s_%s.RData", preds_path, ds, om))
      tr_oracle_labs <- preds
      # Surrogate test labels
      # load(sprintf("%stest_%s_%s.RData", preds_path, ds, om))
      # ts_oracle_labs <- preds
      # 
      df <- data.frame("id"=ids,
                       "Dataset"=ds,
                       "Classes"=length(classes),
                       "Features"=ncol(datos)-1,
                       "Instances"=nrow(datos),
                       "Oracle"=om,
                       "Oracle.output"=predictionsState(tr_oracle_labs, length(classes)))
      # for (sm in models) {
      #   #sm <- "c5.0"
      #   load(sprintf("%stest_%s_%s_%s.RData", preds_path, ds, om, sm))
      #   
      #   df <- cbind(df, data.frame(sm=predictionsState(preds, length(classes))))
      #   colnames(df)[ncol(df)] <- sm
      # }
      csv_control <- rbind(csv_control, df)
    }
    ids <- ids + 1
  }
  
  write.table(x=csv_control[2:nrow(csv_control),], 
              file=paste(analytics_path, "oracles_output.csv", sep=""), 
              row.names = FALSE, sep=",")
}

prediction_status("2_family_identification/files/surrogate_output/")
  
  