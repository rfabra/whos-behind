options( java.parameters = "-Xmx6g" )
options(warn=1)

cp.path_base <- "files/probs/"
cp.ds_path <- "datasets/extension_caepia/"

oracle_surrogate_predictions <- function(cp.path_base, cp.ds_path) {
  
  source("src/2_family_identification/utils/utils.R")
  cp.ds_path <- correct_path(cp.ds_path)
  cp.path_base <- correct_path(cp.path_base)
  
  cp.init <- init_probs(cp.path_base, cp.ds_path)
  
  cp.datasets <- cp.init$datasets#[1:5]
  cp.models <- cp.init$models
  
  cp.nds = length(cp.datasets)
  cp.nmodels <- length(cp.models)
  cp.ids <- 1
  for(cp.ds in cp.datasets) {
    #cp.ds <- "diabetes.csv"
    
    cp.datos <- load_dataset(paste(cp.ds_path, cp.ds, sep=""))
    # cp.datos$Class <- correct_class_names(cp.datos$Class)
    cp.Train <- cp.datos
    
    set.seed(3)
    cp.GridTrain<-generateUnifTest(cp.Train, 100 * (ncol(cp.datos)-1))
    set.seed(4)
    cp.GridTest<-generateUnifTest(cp.Train, 100 * (ncol(cp.datos)-1))
    
    save(cp.GridTrain, file=sprintf("%ssurrogate_datasets/train_seed3_%s.RData", cp.path_base, cp.ds))
    save(cp.GridTest, file=sprintf("%ssurrogate_datasets/test_seed4_%s.RData", cp.path_base, cp.ds))
    cp.i_om <- 0
    for (cp.om in cp.models) {
      # cp.om <- "svmRadialCost_C2_5"
      cp.path_lock <- sprintf("%slocks/%s_%s", cp.path_base, cp.ds, cp.om, smodel = "")
      if (!file.exists(cp.path_lock)) {
        file.create(cp.path_lock)
        cp.i_om <- cp.i_om + 1
        print("---------------------")
        print(paste("Dataset:", cp.ds, "(", cp.ids, "/", cp.nds, "):", cp.ds))
        print(paste("Oracle Model", cp.om, "(", cp.i_om, "/", cp.nmodels, "):", cp.om))
        print("---------------------")
        
        preds <- learn_and_evaluate(cp.Train, cp.GridTrain, cp.om,
                                                 sprintf("%spredictions/train_%s_%s.RData", cp.path_base, cp.ds, cp.om))
        cp.GridTrain$Class <- preds$crisp
        # cp.GridTrain$Class <- probs_to_labels_crisp("files/crisp_proves/", cp.ds, "train", cp.om)
        cp.GridTrain$Class <- factor(cp.GridTrain$Class, levels=unique(cp.GridTrain$Class))
        
        
        preds <- surrogate_labels(cp.GridTrain, cp.GridTest, cp.models, 
                                  sprintf("%spredictions/test_%s_%s", cp.path_base, cp.ds, cp.om))
        
        learn_and_evaluate(cp.Train, cp.GridTest, cp.om,
                                                sprintf("%spredictions/test_%s_%s.RData", cp.path_base, cp.ds, cp.om))
        
        # GridTrain$Class <- factor(GridTrain$Class, levels=unique(GridTrain$Class))
        # GridTest$Class <- factor(GridTrain$Class, levels=unique(GridTrain$Class)))
        
        file.remove(cp.path_lock)
      }
    }
    cp.ids <- cp.ids + 1
  }
  }
