source("src/2_family_identification/probs/compute_predictions.R")
source("src/2_family_identification/probs/predictions_status.R")
source("src/2_family_identification/probs/metafeats.R")
source("src/2_family_identification/probs/identify.R")

cp.path_base <- "files/probs/"
cp.ds_path <- "datasets/extension_caepia/"
# 
# ps.path_base <- "files/family_identification/probs/"
# ps.ds_path <- "datasets/extension_caepia/"

mf.path_base <- "files/probs/"
mf.ds_path <- "datasets/extension_caepia/"

oracle_surrogate_predictions(cp.path_base, cp.ds_path)
metafeats(mf.path_base, mf.ds_path)
maximum_similarity(mf.path_base, mf.ds_path)
metamodel(mf.path_base, mf.ds_path, c("Kappa", "MAE", "Kappa_MAE"))
