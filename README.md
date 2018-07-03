# whos-behind
We developed a methodology to identify the family of a model, given as a black-box oracle.
The file main.py is the entry point of our software.

Given a set of datasets (25 UCI datasets), the model applies 11 learning techniques (belonging to different families) to each dataset,
so that we obtain 275 models.
These are the black-box oracle models who's family we want to identify.


For each UCI dataset, we create a set of artificial examples generated at random following the uniform distribution for each feature.
This artificial examples can later be labelled by the oracle, so that we obtain a labeled dataset that contains the output of the oracle.
We call this dataset the surrogate dataset. 

Given a surrogate dataset (which contains the output of an oracle), we can learn new models that, ideally, will imitate the behaviour of the oracle model.
More concretely, we use the same 11 learning techniques to learn 11 models. We call these models the surrogate models.
We can evaluate each surrogate dataset with each of the surrogate models. So, we can obtain a performance measure per surrogate model.
We chose the Cohen's Kappa score, which measures the degree of agreement between the output of the oracle (labels of the surrogate dataset) and the ouptut of the surrogate models.

At this point, each oracle is characterized by 11 kappa metrics.
To identify the model family, we employed two methods.
First, we consider that the surrogate model with the highest kappa is more likely to belong to the same family as the oracle,
so we assign the label corresponding to the best model family according to the kappa measure.
Second, we represent each oracle with 11 features (each of the obtained kappa), so that we learn a meta-model (SVM) from
all the oracles previously learnt (and evaluated as described above). This meta-model is able to determine the model family of a new oracle model, after processing it to extract the 11 kappa as previously described.
