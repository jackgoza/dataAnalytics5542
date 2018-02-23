package modelGeneration

object IPSettings {
  val setId = ""

  val PATH = "datasets/"
  val INPUT_DIR = PATH + "train/" + setId
  val FEATURES_PATH = PATH + "models/features/" + setId
  val KMEANS_PATH = PATH + "models/clusters/" + setId
  val KMEANS_CENTERS_PATH = PATH + "models/clusterCenters/" + setId
  val HISTOGRAM_PATH = PATH + "models/histograms/" + setId
  val NAIVE_BAYES_PATH = PATH + "models/bayes/" + setId
  val DECISION_TREE_PATH = PATH + "models/decisionTree/" + setId
  val RANDOM_FOREST_PATH = PATH + "models/randomForest/" + setId
}

