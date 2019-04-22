################ Generate Gris.. dangerous if number of features > 9
#setwd("~/ML Family/")
generateGridTest<-function (data,n=5)
{
  
  
  numcols<-ncol(data)-1
  listafeatures<-list()
  for (f in (1:numcols))
  {
    if (!is.factor(data[,f]))
    {
      listafeatures[[names(data)[f]]]<-seq(min(data[,f]),max(data[,f]),length.out=n)
    }
    else
    {
      listafeatures[[names(data)[f]]]<-levels(data[,f])
    }
  }
  
  
  ntest<-expand.grid(listafeatures)
  
  ntest
}

###################### Generate Uniform

generateUnifTest<-function (data,n=1000)
{
  k<-n
  numcols<-ncol(data)-1
  ntest<-data.frame()
  for (f in (1:numcols))
  {
    if (!is.factor(data[,f]))
    {
      if (!is.integer(data[,f]))
      {
        if (f==1) 
        { 
          ntest<-data.frame("kk"=runif(k,min(data[,f]),max(data[,f])))
          names(ntest)[1]<-names(data)[f]
        }
        else {
          #ntest[names(data)[f]] <- rnorm(k,mean = mean(data[,f]),sd = sd(data[,f]))
          ntest[names(data)[f]] <- runif(k,min(data[,f]),max(data[,f]))
        } 
      } else {
        if (f==1) 
        {
          ntest<-data.frame("kk"=round(runif(k,min(data[,f]),max(data[,f]))))
          #ntest<-data.frame("kk"=round(rnorm(k,mean = mean(data[,f]),sd = sd(data[,f])),0))
          names(ntest)[1]<-names(data)[f]
        }
        else {
          ntest[names(data)[f]]<-round(runif(k,min(data[,f]),max(data[,f])))
        }
      }
    }
    else
    {
      if (f==1) 
      {
        ntest<-data.frame("kk"=sample (levels(data[,f]),k, replace=TRUE))
        names(ntest)[1]<-names(data)[f]
      }
      else {
        ntest[[names(data)[f]]]<-sample (levels(data[,f]),k, replace=TRUE)
      }
      ntest[[names(data)[f]]]<-as.factor( ntest[[names(data)[f]]])
      levels( ntest[[names(data)[f]]])<-levels(data[,f])
    }
  }
  ntest
}

#################### Generate uniform test by sampling attribute values from training
generatePriorTest<-function (data,n=1000)
{
  k<-n
  numcols<-ncol(data)-1
  ntest<-data.frame()
  for (f in (1:numcols))
  {
    if (f==1) 
    {
      ntest<-data.frame("kk"=sample(data[,f],k, replace=TRUE))
      names(ntest)[1]<-names(data)[f]
    }
    else {
      ntest[names(data)[f]]<-sample(data[,f],k, replace=TRUE)
    }
  }
  ntest
}

#################### Approximate each feature as a normal distribution, and generate a value from it
generateNormalTest<-function (data,n=1000)
{
  data <- Train
  n <- nrow(Train) * (ncol(datos)-1)
  
  k<-n
  numcols<-ncol(data)-1
  ntest<-data.frame()
  for (f in (1:numcols))
  {
    if (!is.factor(data[,f]))
    {
      if (!is.integer(data[,f]))
      {
        if (f==1) 
        { 
          ntest<-data.frame("kk"=as.numeric(rnorm(k,mean = mean(data[,f]),sd = sd(data[,f]))))
          #ntest<-data.frame("kk"=runif(k,min(data[,f]),max(data[,f])))
          names(ntest)[1]<-names(data)[f]
        }
        else {
          ntest[names(data)[f]] <- as.numeric(rnorm(k, mean = mean(data[,f]),sd = sd(data[,f])))
          #ntest[names(data)[f]]<-runif(k,min(data[,f]),max(data[,f]))
        } 
      } else {
        if (f==1) 
        {
          ntest<-data.frame("kk"= as.integer(round(rnorm(k,mean = mean(data[,f]),sd = sd(data[,f])))))
          #ntest<-data.frame("kk"=round(runif(k,min(data[,f]),max(data[,f]))),0)
          names(ntest)[1]<-names(data)[f]
        }
        else {
          ntest[names(data)[f]]<-as.integer(round(rnorm(k, mean = mean(data[,f]),sd = sd(data[,f]))))
        }
      }
    }
    else
    {
      if (f==1) 
      {
        ntest<-data.frame("kk"=sample (data[,f],k, replace=TRUE))
        names(ntest)[1]<-names(data)[f]
      }
      else {
        ntest[[names(data)[f]]]<-sample (data[,f],k, replace=TRUE)
      }
      ntest[[names(data)[f]]]<-as.factor( ntest[[names(data)[f]]])
      levels( ntest[[names(data)[f]]])<-levels(data[,f])
    }
  }
  ntest
}