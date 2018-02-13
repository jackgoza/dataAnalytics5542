import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils

def main(args: Array[String]) {
  val conf = new SparkConf()
    .setAppName(s"IPApp")
    .setMaster("local[*]")
    .set("spark.executor.memory", "6g")
    .set("spark.driver.memory", "6g")
  val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

  val sc=new SparkContext(sparkConf)

  val images = sc.wholeTextFiles(s"${IPSettings.INPUT_DIR}/*/*.jpg")
