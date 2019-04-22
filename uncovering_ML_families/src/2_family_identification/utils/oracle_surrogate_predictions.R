{
options( java.parameters = "-Xmx6g" )
options(warn=1)

source("common/artificial_2D_dataset.R")
dss <- c(artificial_2D_dataset())



# Load source code
source("common/RunMethods_v4_preproc.R")
source("common/common.R")

# Load datasets
datasets <- list.files("datasets/clean/", pattern=".csv")
datasets <- datasets[!grepl("coltypes", datasets)]

datasets <- datasets[1:10]

nds = length(datasets)

# Load models
#source("1_family_dendrograms/save_load/models_v5.R")
source("2_family_identification/models.R")
nmodels <- length(models)
ids <- 1
}
for(ds in datasets) {
  #ds <- "eucalyptus.csv"
  datos <- load_dataset(sprintf("datasets/clean/%s",ds))
  Train<-datos
  
  
  
  #GridTest<-generateNormalTest(Train, nrow(Train) * (ncol(datos)-1))
  set.seed(3)
  GridTrain<-generateUnifTest(Train, 100 * (ncol(datos)-1))
  set.seed(4)
  GridTest<-generateUnifTest(Train, 100 * (ncol(datos)-1))
  #GridTest <- GridTest[,!(colnames(GridTest) %in% c("X0"))]
  #GridTest <- rbind(GridTest, Train[,1:ncol(Train)-1])
  
  save(GridTrain, file=sprintf("2_family_identification/files/surrogate_datasets/train_%s_seed3.RData", ds))
  save(GridTest, file=sprintf("2_family_identification/files/surrogate_datasets/test_%s_seed4.RData", ds))
  i_om <- 0
  for (om in models)
  {
    #om <- "c5.0"
    i_om <- i_om + 1
     print("---------------------")
     print(paste("Dataset:", ds, "(", ids, "/", nds, "):", ds))
     print(paste("Oracle Model", om, "(", i_om, "/", nmodels, "):", om))
     print("---------------------")
    
    path <- sprintf("2_family_identification/files/surrogate_datasets/%s_%s_oracle_train.RData", ds, om)
    m_lock  <- sprintf("2_family_identification/files/locks/oracle_%s_%s", ds, om)
    if (!file.exists(path)) {
      if (!file.exists(m_lock)) {
        file.create(m_lock)
        
        set.seed(3)
        preds <- runmethods(om, Train, GridTrain);
        print(predictionsState(preds))
        save(preds, file=path)
        
        path <- sprintf("2_family_identification/files/surrogate_datasets/%s_%s_oracle_test.RData", ds, om)
        set.seed(3)
        preds <- runmethods(om, Train, GridTest);
        print(predictionsState(preds))
        save(preds, file=path)
        
        if (file.exists(m_lock)) {file.remove(m_lock)}
      }
      else {
        print(sprintf("Already processed: %s, oracle %s", ds, om))
      }
    }
    
    path <- sprintf("2_family_identification/files/surrogate_datasets/%s_%s_oracle_train.RData", ds, om)
    load(path)
    GridTrain$Class <- preds
    
    path <- sprintf("2_family_identification/files/surrogate_datasets/%s_%s_oracle_test.RData", ds, om)
    load(path)
    GridTest$Class <- preds
    i_sm <- 0
    
    for (sm in models) {
      #sm <- "svmRadialCost_C2_11"
      #sm <- "mlp_3"
      i_sm <- i_sm + 1
      print("---------------------")
      print(paste("Dataset:", ds, "(", ids, "/", nds, "):", ds))
      print(paste("Oracle:", om, "Surrogate Model", sm, "(", i_sm, "/", nmodels, "):", sm))
      print("---------------------")
      
      path <- sprintf("2_family_identification/files/models_predictions/%s_%s_%s_surrogate_test.RData", ds, om, sm)
      m_lock  <- sprintf("2_family_identification/files/locks/surrogate_%s_%s_%s", ds, om, sm)
      if (!file.exists(path)) {
        if (!file.exists(m_lock)) {
          file.create(m_lock)
          
          set.seed(3)
          preds <- runmethods(sm, GridTrain, GridTest);
          print(predictionsState(preds))
          save(preds, file=path)
          
          if (file.exists(m_lock)) {file.remove(m_lock)}
        }
        else {
          print(sprintf("Already processed: %s, oracle %s, surrogate %s", ds, om, sm))
        }
      }
      #else {
      #  load(path)
      #}
    }
  }
  
#  listarestotals[[ds]]<-listares
  ids <- ids + 1
}

#save(listarestotals, file="1_family_dendrograms/files/complete_results.RData")