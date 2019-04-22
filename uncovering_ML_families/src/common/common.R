
load_install <- function(lib_name)
{
  if ((lib_name %in% installed.packages()) == FALSE) 
  {
    install.packages(lib_name)
  }
  require(lib_name, character.only = TRUE)
}

# Load dataset in CSV format
load_dataset <- function(filename) {
  # Load dataset and column type
  dataset <- read.table(filename, sep=",", header=TRUE, check.names=FALSE)
  col_types <- read.table(gsub(".csv","_coltypes.csv", filename))
  
  # Assign column types to dataset
  for (c in 1:nrow(col_types)) {
    if (col_types[c,1] == "factor") {
      dataset[,c] <- as.factor(dataset[,c])
    }
    else if (col_types[c,1] == "integer")
    {
      dataset[,c] <- as.integer(dataset[,c])
    }
    else {
      dataset[,c] <- as.numeric(dataset[,c])
    }
  }
  # Class as factor
  dataset[,ncol(dataset)] <- as.factor(dataset[,ncol(dataset)])
  dataset$Class <- correct_class_names(dataset$Class)
  return(dataset)
}



predictionsState <- function(predictions, nClasses) {
  state <- "OK"
  if (length(predictions) == 0) {
    state <- "NA"
  }
  else {
    if (length(unique(predictions)) == 1) {
      state <- "SINGLE PREDICTION"
    }
    if (nClasses > 2 && length(unique(predictions)) == 2) {
      state <- "2-CLASS PREDICTION"
    }
    if (any(is.na(predictions))) {
      state <- "ERROR"
    }
    else if (all(predictions == FALSE)) {
      state <- "ERROR"
    }
  }
  return(state)
}


predictionsStateProb <- function(predictions) {
  state <- "OK"
  if (length(predictions) == 0) {
    state <- "NA"
  }
  if (any(is.na(predictions))) {
    state <- "ERROR"
  }
  else if (all(predictions == FALSE)) {
    state <- "ERROR"
  }
  state
}

