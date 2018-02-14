object IPSettings {
  val setId = ""

  val PATH = "data/"
  val INPUT_DIR = PATH + "train/" + setId
  val TEST_INPUT_DIR = PATH + "test2/" + setId
  val FEATURES_PATH = PATH + "model/features/" + setId

  val KMEANS_PATH = PATH + "model/clusters/" + setId
  val KMEANS_CENTERS_PATH = PATH + "model/clusterCenters/" + setId
  val HISTOGRAM_PATH = PATH + "model/histograms/" + setId
  val NAIVE_BAYES_PATH = PATH + "model/bayes/" + setId
  val DECISION_TREE_PATH = PATH + "model/decisionTree/" + setId
  val RANDOM_FOREST_PATH = PATH + "model/randomForest/" + setId
}

