object IPSettings {
  val setId = ""

  val PATH = "../data/"
  val INPUT_DIR = PATH + "train/" + setId
  val TEST_INPUT_DIR = PATH + "test/" + setId
  val FEATURES_PATH = PATH + "models/features/" + setId
  val KMEANS_PATH = PATH + "models/clusters/" + setId
  val KMEANS_CENTERS_PATH = PATH + "models/clusterCenters/" + setId
  val HISTOGRAM_PATH = PATH + "models/histograms/" + setId
  val NAIVE_BAYES_PATH = PATH + "models/naiveBayes/" + setId
  val DECISION_TREE_PATH = PATH + "models/decisionTree/" + setId
  val RANDOM_FOREST_PATH = PATH + "models/randomForest/" + setId
}

