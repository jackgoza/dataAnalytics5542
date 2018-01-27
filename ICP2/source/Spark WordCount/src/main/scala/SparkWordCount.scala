import org.apache.spark.{SparkContext, SparkConf}

object SparkWordCount {

  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    val input=sc.textFile("input")

    val wc=input.flatMap(line=>{line.split(" ")}).map(word=>(word.charAt(0).toUpper, word)).cache()

    val output=wc.reduceByKey(_ + ", " + _)

    output.coalesce(1).saveAsTextFile("moreWords")

    val o=output.collect()

    var s:String="Words:Count \n"
    o.foreach{case(word,count)=>{

      s+=word+" : "+count+"\n"

    }}

  }

}
