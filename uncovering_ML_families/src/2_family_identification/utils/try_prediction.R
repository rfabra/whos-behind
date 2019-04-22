

source("src/2_family_identification/utils/utils.R")
source("src/common/common.R")
source("src/common/queries.R")

path_base <- "files/prob_proves/"
ds_path <- "datasets/extension_caepia/"  

#init_train_test(ds_path)
init_probs(path_base, ds_path)
ds <- "car.csv"
om <- "glmnet"
sm <- "c5.0"

datos <- load_dataset(sprintf("datasets/extension_caepia/%s",ds))
datos$Class <- as.factor(sapply(datos$Class, function(x) {
  if (x == '+') return('PLUS') else if (x == '-') return('MINUS') else return(make.names(x))}))
# cp.Train<-cp.datos

Train<-datos

set.seed(3)
GridTrain<-generateUnifTest(Train, 100 * (ncol(datos)-1))
set.seed(4)
GridTest<-generateUnifTest(Train, 100 * (ncol(datos)-1))

preds <- learn_and_evaluate(Train, GridTrain, om)


GridTrain$Class <- preds$crisp
GridTrain$Class <- factor(GridTrain$Class, levels = unique(GridTrain$Class))
predictionsStateProb(GridTrain$Class)
# predictionsState(GridTrain$Class)
#GridTrain$Class <- probs_to_labels(GridTrain$Class)
predictionsState(GridTrain$Class, length(unique(Train$Class)))

preds <- learn_and_evaluate(GridTrain, GridTest, sm)
GridTest$Class <- preds$crisp 
predictionsStateProb(GridTest$Class)
GridTest$Class <- probs_to_labels(GridTest$Class)
predictionsState(GridTest$Class, length(unique(Train$Class)))

#GridTest$Class <- learn_and_evaluate(Train, GridTest, sm)

#GridTest$Class <- factor(GridTest$Class, levels=unique(Train$Class))
#GridTest[[sm]] <- factor(GridTest[[sm]], levels=unique(Train$Class))

conf_mat <- caret::confusionMatrix(GridTest$Class, GridTest[[sm]], mode="everything")
kappas[[sm]] <- conf_mat$overall["Kappa"]
                 