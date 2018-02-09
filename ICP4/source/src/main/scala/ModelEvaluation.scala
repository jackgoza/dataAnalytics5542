import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

object ModelEvaluation {

  val conf = new SparkConf()
    .setAppName(s"IPApp")
    .setMaster("local[*]")
    .set("spark.executor.memory", "6g")
    .set("spark.driver.memory", "6g")
  val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

  val sc=new SparkContext(sparkConf)

  val predicted = Seq.newBuilder.+=((1.0, 0.0))
  predicted.+=((1.0,1.0))
  predicted.+=((0.0,0.0))
  predicted.+=((1.0,1.0))
  predicted.+=((0.0,1.0))
  predicted.+=((0.0,0.0))
  predicted.+=((0.0,0.0))
  predicted.+=((1.0,1.0))
  predicted.+=((1.0,0.0))



  def main(args: Array[String]): Unit = {
    evaluateModel(sc.parallelize(predicted.result()))
  }


  def evaluateModel(predictionAndLabels: RDD[(Double, Double)]) = {
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val cfMatrix = metrics.confusionMatrix
    println(" |=================== Confusion matrix ==========================")
    println(cfMatrix)
    println(metrics.fMeasure)


    printf(
      s"""
         |=================== Confusion matrix ==========================
         |          | %-15s                     %-15s
         |----------+----------------------------------------------------
         |Actual = 0| %-15f                     %-15f
         |Actual = 1| %-15f                     %-15f
         |===============================================================
         """.stripMargin, "Predicted = 0", "Predicted = 1",
      cfMatrix.apply(0, 0), cfMatrix.apply(0, 1), cfMatrix.apply(1, 0), cfMatrix.apply(1, 1))

    println("\nACCURACY " + ((cfMatrix(0,0) + cfMatrix(1,1))/(cfMatrix(0,0) + cfMatrix(0,1) + cfMatrix(1,0) + cfMatrix(1,1))))


    cfMatrix.toArray

    val fpr = metrics.falsePositiveRate(0)
    val tpr = metrics.truePositiveRate(0)

    println(
      s"""
         |False positive rate = $fpr
          |True positive rate = $tpr
     """.stripMargin)

    val m = new BinaryClassificationMetrics(predictionAndLabels)
    println("PR " + m.areaUnderPR())
    println("AUC " + m.areaUnderROC())
  }
}
