{
options(java.parameters = "-Xmx6g" )
options(warn=1)

# Maximum number of features and instances allowed per dataset
max_feats <- 500
max_instances <- 12000

# Correct class attribute, name and position
correct_load <- function (ds_name, ds_data)
{
  if (ds_name == "Click_prediction_small.arff" ||
      ds_name == "JapaneseVowels.arff" ||
      ds_name == "lung-cancer.arff" ||
      ds_name == "monks-problems-1.arff" ||
      ds_name == "monks-problems-2.arff" ||
      ds_name == "monks-problems-3.arff") {
    ds_data <- cbind(ds_data[,2:ncol(ds_data)], ds_data[,1])
    colnames(ds_data)[ncol(ds_data)]<-"Class"
  }
  if (ds_name == "splice.arff" || ds_name == "synthetic_control.arff") {
    # El primer atribut es un identificador de la instància.
    # Això probablement mareje, ja que pot prendre molts valors possibles
    ds_data <- ds_data[,2:ncol(ds_data)]
  }
  colnames(ds_data)[ncol(ds_data)]<-"Class"
  
  ds_data
}


enough_feats <- function(datos, remaining_feats, csv_data) {
  # Check we have not lost 25% of feats
  nFeats <- ncol(datos)-1
  nFeats_clean <- ncol(remaining_feats)-1
  
  enough <- FALSE
  if (nFeats_clean/nFeats < 0.75) {
    csv_data[["selected?"]][[ids]] <- "NO"
    csv_data[["reason"]][[ids]] <- "Lost more than 25% of features after cleanup"
    print(sprintf("NOT SELECTED: %s", csv_data[["reason"]][[ids]]))
  } else {
    csv_data[["selected?"]][[ids]] <- "YES"
    csv_data[["reason"]][[ids]] <- ""
    enough <- TRUE
  }
  retList <- list(enough=enough, csv_data=csv_data)
  retList
}

enough_examples <- function(datos, remaining_examples, csv_data) {
  # Check we have not lost 25% of examples
  # Minimum threshold (%) of examples to keep to preserve the dataset
  min_row_thr <- 0.70
  
  nInstances <- nrow(datos)
  nInstances_clean <- nrow(remaining_examples)
  
  enough <- FALSE
  if (nInstances_clean/nInstances < min_row_thr) {
    csv_data[["selected?"]][[ids]] <- "NO"
    csv_data[["reason"]][[ids]] <- sprintf("Lost more than %s of examples after cleanup",min_row_thr*100)
    print(sprintf("NOT SELECTED: %s", csv_data[["reason"]][[ids]]))
  } else {
    csv_data[["selected?"]][[ids]] <- "YES"
    csv_data[["reason"]][[ids]] <- ""
    enough <- TRUE
  }
  retList <- list(enough=enough, csv_data=csv_data)
  retList
}

remove_missing_feats <- function(feats, csv_data, ids) {
  nFeats <- ncol(feats)
  n_samples_per_feat <- data.frame(apply(feats, 2, length))
  n_NA_feats <- apply(feats, 2, FUN=function(x) {sum(!is.na(x))})
  prop_na_samples <- n_NA_feats/n_samples_per_feat
  
  # Remove features for which at least 25 of examples have missing value
  remaining_feats <- feats[,apply(prop_na_samples, 1, function(x) {x>=0.75})]
  csv_data[["#missing feats (> 25% of examples)"]][[ids]] <- csv_data[["#missing feats (> 25% of examples)"]][[ids]] + (nFeats - ncol(remaining_feats))
  csv_data[["missing feats %"]][[ids]] <- round((csv_data[["#missing feats (> 25% of examples)"]][[ids]]/csv_data[["#feats"]][[ids]]) * 100, 2)
  
  retlist <- list(data=remaining_feats, csv_data=csv_data)
  retlist
}

remove_0_var_feats <- function(remaining_feats, csv_data, ids) {
  # Remove 0 variance features
  var0_feats_num = apply(remaining_feats, 2, var, na.rm=TRUE) %in%c(0)
  remaining_feats <- remaining_feats[,!var0_feats_num]
  var0_feats_disc = apply(remaining_feats, 2, function(x){length(unique(x))==1})
  remaining_feats <- remaining_feats[,!var0_feats_disc]
  
  csv_data[["#feats_0_var"]][[ids]] <- csv_data[["#feats_0_var"]][[ids]] + length(var0_feats_num[var0_feats_num == TRUE]) + length(var0_feats_disc[var0_feats_disc == TRUE])
  csv_data[["0_var missing %"]][[ids]] <- round((csv_data[["#feats_0_var"]][[ids]]/csv_data[["#feats"]][[ids]]) * 100, 2)
  
  csv_data[["#remaining feats"]][[ids]] <- ncol(remaining_feats)
  csv_data[["remaining feats %"]][[ids]] <- round((csv_data[["#remaining feats"]][[ids]]/csv_data[["#feats"]][[ids]]) * 100, 2)
  
  retlist <- list(data=remaining_feats, csv_data=csv_data)
  retlist
}

remove_missing_instances <- function(remaining_examples, csv_data, ids) {
  # Remove rows with missing values
  na_examples <- complete.cases(remaining_examples)
  csv_data[["missing examples"]][[ids]] <- csv_data[["missing examples"]][[ids]] + length(na_examples[na_examples == FALSE])
  csv_data[["missing examples %"]][[ids]] <- round((csv_data[["missing examples"]][[ids]]/csv_data[["#instances"]][[ids]]) * 100, 2)
  remaining_examples<-remaining_examples[na_examples, ]
  
  retlist <- list(data=remaining_examples, csv_data=csv_data)
  retlist
}

remove_duplicated_instances <- function(remaining_examples, csv_data, ids) {
  feats <- remaining_examples[,1:ncol(remaining_examples)-1]
  dup_idx <- (duplicated(feats) | duplicated(feats, fromLast=TRUE))
  dup_data <- feats[dup_idx,]
  dups <- unique(dup_data)
  NDUPS <- nrow(dups)
  n_dup <- 0
  if (NDUPS > 250) {
    csv_data[["selected?"]][[ids]] <- "NO"
    csv_data[["reason"]][[ids]] <- sprintf("More than 250 examples (potentially) duplicated: (%s/%s, %s per cent", NDUPS, nrow(remaining_examples), NDUPS/nrow(remaining_examples))
    print(csv_data[["reason"]][[ids]] <- sprintf("Too many duplicated instances to process: (%s/%s, %s per cent", NDUPS, nrow(remaining_examples), NDUPS/nrow(remaining_examples)))
  } 
  else if (nrow(dups) > 0) {
    for (d in 1:nrow(dups)) {
      print(sprintf("Analyzing duplicates (%s/%s)",d, NDUPS))
      dup <- dups[d,]
      dup_rows <- apply(feats, 1, function(x){ all(x == dup) })
      if (length(unique(remaining_examples[dup_rows,ncol(remaining_examples)])) > 1) {
        feats <- feats[!dup_rows,]
        remaining_examples <- remaining_examples[!dup_rows,]
        n_dup <- n_dup + length(dup_rows[dup_rows])
      }
    }
  }
  
  csv_data[["duplicated examples"]][[ids]] <- csv_data[["duplicated examples"]][[ids]] + n_dup
  csv_data[["duplicated examples %"]][[ids]] <- round((n_dup/csv_data[["#instances"]][[ids]]) * 100, 2)
  csv_data[["#remaining examples"]][[ids]] <- nrow(remaining_examples)
  
  retlist <- list(data=remaining_examples, csv_data=csv_data)
  retlist
}


# load source code
library(farff)
library(entropy)
library(plyr)
source("src/common/common.R")

datasets <- list.files('datasets/extension_caepia_arff', pattern=".arff")
nds = length(datasets)

# Trobar el nombre màxim de classes automàticament
# max_classes = 0
# 
# for (ds in datasets) {
#   datos <- readARFF(paste("datasets/all/",ds,sep=""))
#   # Correct the load for specific datasets
#   datos <- correct_load(ds, datos)
#   datos$Class <- as.factor(datos$Class)
#   classes = unique(datos$Class)
#   nClasses <- length(classes)
#   if (nClasses > max_classes) {
#     max_classes <- nClasses
#   }
# }

# Més ràpid a pinyó fixe
max_classes <- 100

# Initialize CSV data
header <- c(
  "Dataset", 
  "#classes", 
  "#feats",          
  "#missing feats (> 25% of examples)", 
  "missing feats %",         
  "#feats_0_var", 
  "0_var missing %",
  "#remaining feats", 
  "remaining feats %",
  "#instances", 
  "missing examples", 
  "missing examples %",
  "duplicated examples",
  "duplicated examples %",
  "#remaining examples",
  "#remaining classes",
  "class entropy ratio")

header <- c(header, paste("class", 1:max_classes))

header <- c(header,"selected?", "reason")

csv_data <- data.frame(matrix(nrow=nds, ncol=length(header)))
colnames(csv_data) <- header
csv_data <- data.frame(apply(csv_data, c(1,2), function(x) {if (is.na(x)){x=0}}))
colnames(csv_data) <- header

capture.output(
for (ds in datasets) {
  datos <- readARFF(paste("datasets/extension_caepia_arff/",ds,sep=""))
  feats.num <- sum(sapply(names(datos[,1:ncol(datos)-1]), function(x) {is.numeric(datos[,x])}))
  feats.disc <- sum(sapply(names(datos[,1:ncol(datos)-1]), function(x) {is.factor(datos[,x])}))
  cat(paste(ds, feats.num, feats.disc, sep=","))
  cat("\n")
}, file="/tmp/foo.csv"
)
#Start cleaning procedure
ids <- 0
}

for (ds in datasets) {
  {
  #ds <- "ada_agnostic.arff"
  ids <- ids + 1
  print(paste("---- Dataset (", ids, "/", nds, "): ", ds))
  datos <- readARFF(paste("datasets/extension_caepia_arff/",ds,sep=""))
  # Correct the load for specific datasets
  datos <- correct_load(ds, datos)
  datos$Class <- as.factor(datos$Class)
  classes = unique(datos$Class)
  
  #remove class factors with no data
  datos$Class <- as.factor(as.character(datos$Class))
  
  # Name
  csv_data[["Dataset"]][[ids]] <- gsub('.arff', '', ds)
  
  # Num. classes, instances and features
  nClasses <- length(classes)
  nInstances <- nrow(datos)
  nFeats <- ncol(datos) - 1
  csv_data[["#classes"]][[ids]] <- nClasses
  csv_data[["#instances"]][[ids]] <- nInstances
  csv_data[["#feats"]][[ids]] <- nFeats
  
  # Check the maximum number of instances and features
  # if (nInstances > max_instances) {
  #   csv_data[["selected?"]][[ids]] <- "NO"
  #   csv_data[["reason"]][[ids]] <- sprintf("More than %s instances", max_instances)
  #   print(sprintf("NOT SELECTED: %s", csv_data[["reason"]][[ids]]))
  #   next
  # }
  # if (nFeats > max_feats) {
  #   csv_data[["selected?"]][[ids]] <- "NO"
  #   csv_data[["reason"]][[ids]] <- sprintf("More than %s features", max_feats)
  #   print(sprintf("NOT SELECTED: %s", csv_data[["reason"]][[ids]]))
  #   next
  # }
  prev_data <- data.frame()
  current_data <- datos[]
  }
  
  anotherLoop <- TRUE
  while (anotherLoop) {
    prev_data <- current_data[]
    nFeats <- ncol(current_data) - 1
    
    # Missing features
    feats <- current_data[,1:nFeats]
    res <- remove_missing_feats(feats, csv_data, ids)
    remaining_feats <- res$data
    csv_data <- res$csv_data
    
    #res <- enough_feats(datos, remaining_feats, csv_data)
    csv_data <- res$csv_data
    # if (!res$enough) {
    #   break
    # }
    
    # 0 variance features
    res <- remove_0_var_feats(remaining_feats, csv_data, ids)
    remaining_feats <- res$data
    csv_data <- res$csv_data
    #nFeats <- ncol(remaining_feats)
    
    # res <- enough_feats(datos, remaining_feats, csv_data)
    csv_data <- res$csv_data
    # if (!res$enough) {
    #   break
    # }
    # Tornem a afegir les classes
    remaining_examples <- cbind(remaining_feats, current_data[,nFeats + 1])
    colnames(remaining_examples)[ncol(remaining_examples)] <- "Class"
    
    # Missing examples 
    res <- remove_missing_instances(remaining_examples, csv_data, ids)
    remaining_examples <- res$data
    csv_data <- res$csv_data
    
    # res <- enough_examples(datos, remaining_examples, csv_data)
    csv_data <- res$csv_data
    # if (!res$enough) {
    #   break
    # }
    
    # duplicated examples
    res <- remove_duplicated_instances(remaining_examples, csv_data, ids)
    remaining_examples <- res$data
    csv_data <- res$csv_data
    
    # res <- enough_examples(datos, remaining_examples, csv_data)
    csv_data <- res$csv_data
    # if (!res$enough) {
    #   break
    # }
    current_data <- remaining_examples
    
    nClasses <- length(unique(datos$Class))
    nClasses_clean <- length(unique(current_data$Class))
    if (nClasses != nClasses_clean) {
      csv_data[["selected?"]][[ids]] <- "NO"
      csv_data[["reason"]][[ids]] <- sprintf("Some classes are lost: from %s to %s", nClasses, nClasses_clean)
      print(sprintf("NOT SELECTED: %s", csv_data[["reason"]][[ids]]))
      break
    }
    if (csv_data[["selected?"]][[ids]] == "NO") {
      print(csv_data[["reason"]][[ids]])
      break
    }
    
    if (nrow(prev_data) == nrow(current_data) && ncol(prev_data) == ncol(current_data)) {
      anotherLoop <- all(ifelse(prev_data == current_data, FALSE, TRUE))
    }
    csv_data[["selected?"]][[ids]] <- "YES"
    csv_data[["reason"]][[ids]] <- ""
  }
  if (csv_data[["selected?"]][[ids]] == "YES") {
    # save_dataset(current_data, gsub(".arff", ".csv", sprintf("datasets/caepia_clean/%s", ds)))
    
    # Compute class imbalance
    ic <- 1
    class_prob = c()
    balanced_labels <- c()
    for (c in classes) {
      classC <- apply(current_data, 1, function(x) { x["Class"] == c})
      nc <- nrow(current_data[classC,])
      prop <- nc/csv_data[["#remaining examples"]][[ids]]
      class_prob <- c(class_prob, prop)
      csv_data[[sprintf("class %d", ic)]][[ids]] <- prop
      balanced_labels <- c(balanced_labels, rep(c, nrow(current_data)/nClasses))
      ic <- ic + 1
    }
    csv_data[["class entropy ratio"]][[ids]] <- entropy(count(current_data$Class)$freq)/entropy(count(balanced_labels)$freq)
  }
  
  #write.table(x=csv_data, file="1_family_dendrograms/files/analytics/datasets_clean_test.csv", row.names = FALSE, col.names = TRUE, sep=",")
}

write.table(x=csv_data, file="datasets/extension_caepia_clean.csv", row.names = FALSE, col.names = TRUE, sep=",")
