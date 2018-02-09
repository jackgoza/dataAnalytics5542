import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

object ConfusionMatrix {

  val conf = new SparkConf()
    .setAppName(s"IPApp")
    .setMaster("local[*]")
    .set("spark.executor.memory", "6g")
    .set("spark.driver.memory", "6g")
  val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

  val sc=new SparkContext(sparkConf)



  def main(args: Array[String]): Unit = {

  }
}
