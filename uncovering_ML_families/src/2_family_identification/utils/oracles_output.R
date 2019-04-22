options( java.parameters = "-Xmx6g" )
options(warn=1)

oracles_output <- function(path_base) {
  source("2_family_identification/utils.R")
  ds_path <- correct_path("datasets/clean/")
  init <- init_train_test(ds_path)
  
  datasets <- init$datasets
  models <- init$models
  
  # datasets <- datasets[1:10]
  # models <- c("c5.0", "mlp_3", "svmRadialCost_C2_5")
  
  nds = length(datasets)
  nmodels <- length(models)
  ids <- 1
  
  for(ds in datasets) {
    # ds <- "diabetes.csv"
    
    datos <- load_dataset(paste(ds_path, ds, sep=""))
    Train<-datos
    
    set.seed(3)
    GridTrain<-generateUnifTest(Train, 100 * (ncol(datos)-1))
    set.seed(4)
    GridTest<-generateUnifTest(Train, 100 * (ncol(datos)-1))
    
    save(GridTrain, file=sprintf("%ssurrogate_datasets/train_seed3_%s.RData", path_base, ds))
    save(GridTest, file=sprintf("%ssurrogate_datasets/test_seed4_%s.RData", path_base, ds))
    i_om <- 0
    for (om in models) {
      # om <- "c5.0"
      path_lock <- sprintf("%slocks/%s_%s", path_base, ds, om)
      if (!file.exists(path_lock)) {
        file.create(path_lock)
        i_om <- i_om + 1
        print("---------------------")
        print(paste("Dataset:", ds, "(", ids, "/", nds, "):", ds))
        print(paste("Oracle Model", om, "(", i_om, "/", nmodels, "):", om))
        print("---------------------")
        
        GridTrain$Class <- learn_and_evaluate(Train, GridTrain, om,
                                              sprintf("%spredictions/train_%s_%s.RData", path_base, ds, om)
        )
        # GridTest$Class <- learn_and_evaluate(Train, GridTest, om,
        #                                      sprintf("%spredictions/test_%s_%s.RData", path_base, ds, om)
        # )
        # preds <- surrogate_labels(GridTrain, GridTest, models, 
        #                           sprintf("%spredictions/test_%s_%s", path_base, ds, om)
        # )
        file.remove(path_lock)
      }
    }
    ids <- ids + 1
  }
}

oracles_output("2_family_identification/files/surrogate_output/")