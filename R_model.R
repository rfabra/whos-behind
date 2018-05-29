library("caret")
library("RcppCNPy")
library("randomForest")
library("MASS")
library("C50")
library("kernlab")
library("klaR")
library("class")
library("pls")
library("earth")
library("RWeka")
library("naivebayes")
library("earth")
library("glmnet")

#options(error=traceback)
options(warn=1)

args = commandArgs(trailingOnly=TRUE)

method <- args[1]
tr_file <- args[2]
tr_l_file <- args[3]
ts_file <- args[4]
pred_file <- args[5]

print("------------------------------")
print("INPUT:")
print(paste("Method:", method))
print(paste("Training file:", tr_file))
print(paste("Training lables file:", tr_l_file))
print(paste("Test file:", ts_file))
print(paste("Pred file:", pred_file))
print("------------------------------")


tr <- npyLoad(tr_file)
#tr_l <- t(read.table(tr_l_file, sep="\n"))
tr_l <- read.table(tr_l_file, sep="\n")
ts <- npyLoad(ts_file)



feat_names  <- c(sprintf("f%02d", seq(1,ncol(tr))))

colnames(tr) <- feat_names
colnames(ts) <- feat_names
colnames(tr_l) <- c("class")
labels = unique(tr_l)

#print(labels[[1,1]])
#stop("----")



#tr$class <- tr_l

#str(tr)
#str(tr_l)

#eapply(.GlobalEnv,typeof)

if (method == "RDA") {
	#ERROR <-  tryCatch(model <- qda(x=tr[,feat_names], as.factor(tr_l$class), error = function(e) {return(TRUE)}))
	#ERROR <-  tryCatch(model <- lda(class ~ ., data = tr, error = function(e) {return(TRUE)}))
	ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "rda"), error = function(e) {return(TRUE)})
	print("Learned model")

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		print("Predicted")
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}
if (method == "RF") {
	#ERROR <-  tryCatch(model <- randomForest(x=tr, y=as.factor(tr_l), error = function(e) {return(TRUE)}))
	#ERROR <-  tryCatch(model <- randomForest(x=tr[,feat_names], y=as.factor(tr_l$class), error = function(e) {return(TRUE)}))
	ERROR <-  tryCatch(model <- randomForest(x=tr[,feat_names], y=as.factor(tr_l$class)), error = function(e) {return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}
if (method == "C5.0") {
	#ERROR <-  tryCatch(model <- C5.0(x=tr[,feat_names], y=as.factor(tr_l$class), error = function(e) {return(TRUE)}))
	ERROR <-  tryCatch(model <- C5.0(x=tr[,feat_names], y=as.factor(tr_l$class)), error = function(e) {return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}

if (method == "JRIP") {
	#ERROR <-  tryCatch(model <- C5.0(x=tr[,feat_names], y=as.factor(tr_l$class)), error = function(e) {return(TRUE)})
	training <- data.frame(cbind(data.frame(tr), data.frame(tr_l)))
	ERROR <-  tryCatch(model <- JRip(class ~ ., data = training), error = function(e) {print(e); return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}

if (method == "SVM_GAUSSIAN") {
	#ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), tuneGrid=data.frame(C=1), method = "svmRadialCost", error = function(e) {return(TRUE)}))
	ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), tuneGrid=data.frame(C=1), method = "svmRadialCost"), error = function(e) {return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}
if (method == "MLP") {
	#ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "nnet", trace = FALSE, error = function(e) {return(TRUE)}))
	#ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "nnet", trace = FALSE), error = function(e) {return(TRUE)})
	ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "mlpML", trControl = trainControl(method="none")), error = function(e) {print(e); return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}
if (method == "NB") {
	#ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "nb", error = function(e) {return(TRUE)}))
	ERROR <-   tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "nb"), error = function(e) {return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}

if (method == "NAIVE BAYES") {
	#ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "nb", error = function(e) {return(TRUE)}))
	#ERROR <-   tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "naive_bayes"), error = function(e) {return(TRUE)})
	ERROR <-  tryCatch(model <- naive_bayes(x=tr[,feat_names], y=as.factor(tr_l$class)), error = function(e) {return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		print("NAIVE BAYES PREDICTION")
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		#preds<- predict(model, ts)
		print(preds)
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}

if (method == "KNN") {
	#ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "knn", error = function(e) {print("Handling exception"); return(TRUE)}))
	ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "knn", tuneGrid=data.frame(k = 5)), error = function(e) {return(TRUE)})
	#ERROR <-  tryCatch(model <- knn(x=tr[,feat_names], y=as.factor(tr_l$class), tuneGrid=data.frame(k = 1), error = function(e) {return(TRUE)}))
	#ERROR <-  tryCatch(model <- IBk(x=tr[,feat_names], y=as.factor(tr_l$class), error = function(e) {return(TRUE)}))
	#training <- data.frame(cbind(data.frame(tr), data.frame(tr_l)))
	#ERROR <-  tryCatch(model <- IBk(class ~., data = training, control = Weka_control(K=1)), error = function(e) {return(TRUE)})
	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		print("Starting prediction...")
		preds <- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		print("Predicting OK")
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}
if (method == "GLMNET") {
	
	#ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "glm", error = function(e) {return(TRUE)}))
	ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "glmnet"), error = function(e) {print(e); return(TRUE)})
	#print(as.factor(tr_l$class))
	#ERROR <-  tryCatch(model <- glmnet(x=tr[,feat_names], y=as.factor(tr_l$class)), family="gaussian", error = function(e) {print(e); return(TRUE)})
	#ERROR <-  model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "glm")

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}
if (method == "SIMPLS") {
	ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "simpls"), error = function(e) {return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}
if (method == "PMR") {
	ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "multinom", trace = FALSE), error = function(e) {return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}
if (method == "PLR") {
	ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "plr", trace = FALSE), error = function(e) {return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}
if (method == "MARS") {
	#ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "gcvEarth"), error = function(e) {return(TRUE)})
	ERROR <-  tryCatch(model <- caret::train(x=tr[,feat_names], y=as.factor(tr_l$class), method = "gcvEarth"), error = function(e) {return(TRUE)})
	#ERROR <-  tryCatch(model <- earth(x=tr[,feat_names], y=as.factor(tr_l$class)), error = function(e) {return(TRUE)})

	if (is.logical(ERROR)) {
		print("Is logical error")
		preds <- rep(FALSE, nrow(ts))

	} else {
		preds<- tryCatch(predict(model, ts), error = function(e) {print(e); return(rep(FALSE, nrow(ts)))})
		if (is.list(preds)) {
			preds <- preds$class
		}
	}
}
#print("Predicted:")
#print(preds)

#npySave(pred_file, c(preds))

write.table(preds, pred_file, row.names=FALSE, col.names=FALSE, quote=FALSE)
