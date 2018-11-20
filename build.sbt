name := "ML"

version := "1.0"

scalaVersion := "2.11.8"


libraryDependencies ++= Seq(
 "org.apache.spark" %% "spark-core" % "2.3.0" % "provided",
 "org.apache.spark" %% "spark-sql" % "2.3.0" % "provided",
 "org.apache.spark" %% "spark-mllib" % "2.3.0" % "provided",
 "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1",
 "org.deeplearning4j" %% "dl4j-spark" % "0.9.1_spark_2",
 "org.nd4j" % "nd4j-native-platform" % "0.9.1",
 "org.datavec" % "datavec-api" % "0.9.1",
)


// stays inside the sbt console when we press "ctrl-c"
Compile / run / fork := true
Global / cancelable := true


assemblyMergeStrategy in assembly := {
 case PathList("javax", "servlet", xs @ _*)         => MergeStrategy.first
 case PathList("org","apache","spark","unused","UnusedStubClass.class") => MergeStrategy.discard
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case PathList(ps @ _*) if ps.last endsWith ".MF"   => MergeStrategy.discard
 case "application.conf"                            => MergeStrategy.concat
 case "unwanted.txt"                                => MergeStrategy.discard
 case x => MergeStrategy.first
}
