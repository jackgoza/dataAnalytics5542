import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types._

object LabOneMain {

  var ratingSchema = new StructType(Array(
    StructField("UserId", IntegerType, true),
    StructField("ItemId", IntegerType, true),
    StructField("Rating", IntegerType, true),
    StructField("Timestamp", IntegerType, true)))

  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setAppName("LabOneMain").setMaster("local[*]")

    val sc = new SparkContext(sparkConf)

    val sqlContext = new SQLContext(sc)

    val input = sqlContext.read.format("com.databricks.spark.csv"). // Use "csv" regardless of TSV or CSV.
      option("header", "false"). // Does the file have a header line?
      option("delimiter", "\t"). // Set delimiter to tab or comma.
      schema(ratingSchema). // Schema that was built above.
      load("ml-100k/u.data")

    val wc=input.map(row => (row.getAs[Int]("UserId"), 1)).cache()

    val output=wc.reduceByKey(_ + _).filter(_._2 > 25)

    output.coalesce(1).saveAsTextFile("users")

//    input.registerTempTable("user_ranks")
//
//    val userRanks = sqlContext.sql("SELECT * FROM user_ranks GROUP BY UserId HAVING COUNT(DISTINCT Timestamp) > 20")
//
//    userRanks.show(30)


    //    val wc=input.flatMap(line=>{line.split(" ")}).map(word=>(word.charAt(0).toUpper, word)).cache()
    //
    //    val output=wc.reduceByKey(_ + ", " + _)
    //
    //    output.coalesce(1).saveAsTextFile("moreWords")
    //
    //    val o=output.collect()
    //
    //    var s:String="Words:Count \n"
    //    o.foreach{case(word,count)=>{
    //
    //      s+=word+" : "+count+"\n"
    //
    //    }}

  }

}
