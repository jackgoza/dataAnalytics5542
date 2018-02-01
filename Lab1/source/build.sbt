import sbt.Keys._

lazy val root = (project in file(".")).
  settings(
    name := "LabOneMain",
    version := "1.0",
    scalaVersion := "2.11.8",
    mainClass in Compile := Some("LabOneMain")
  )

exportJars := true
fork := true

val meta = """META.INF(.)*""".r

// additional libraries
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.1" % "provided",
  "org.apache.spark" %% "spark-streaming" % "1.6.1",
  "org.apache.spark" %% "spark-mllib" % "1.6.1",
  "com.databricks" % "spark-csv_2.11" % "1.2.0"
)

