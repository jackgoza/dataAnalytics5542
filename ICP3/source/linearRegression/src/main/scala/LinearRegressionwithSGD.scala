import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
//import org.apache.log4j.{Level, Logger}
/**
  * Created by Manikanta on 1/30/2017.
  */
object LinearRegressionwithSGD {

  def main(args: Array[String]): Unit ={


//    System.setProperty("hadoop.home.dir","C:\\Users\\Manikanta\\Documents\\UMKC Subjects\\PB\\hadoopforspark");

    val sparkConf = new SparkConf().setAppName("LinearRegression").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    // Load and parse the data
    val data = sc.textFile("data/3D_spatial_network.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(Array(parts(1).toDouble, parts(2).toDouble, parts(3).toDouble)))
    }.cache()

    parsedData.take(10).foreach(f=>println(f))

    // Split data into training (95%) and test (5%).
    val Array(training, test) = parsedData.randomSplit(Array(0.70, 0.30))

    // Building the model
    val numIterations = 10000
    val stepSize = 0.00000001
    val model = LinearRegressionWithSGD.train(training, numIterations, stepSize)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = training.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    println("3D Road Linear Regression 50 iteration 70/30 train/test")
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("training Mean Squared Error = " + MSE)

    // Evaluate model on test examples and compute training error
    val valuesAndPreds2 = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE2 = valuesAndPreds2.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("test Mean Squared Error = " + MSE2)

    // Save and load model
    model.save(sc, "data/LinearRegressionWith3dRoadModel")
    // val sameModel = LinearRegressionModel.load(sc, "data/LinearRegressionWithSGDModel")
  }

}
