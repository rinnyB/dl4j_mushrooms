package aw

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline

import org.apache.spark.ml.linalg.{SparseVector, DenseVector}

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}

import org.deeplearning4j.spark.api.TrainingMaster
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.eval.Evaluation;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j

object classification_example {

 def main(args: Array[String]): Unit = {

   val sconf = new SparkConf().setAppName("ML")
     .setMaster("local[2]")
     .set("spark.driver.memory", "2g")
     .set("spark.executor.memory", "2g")
   val sc = new SparkContext(sconf)
   sc.setLogLevel("ERROR")
   val spark = SparkSession.builder.master("local").appName("ML").getOrCreate()
  
   // field names are missing in csv
   // it is taken from dataset description
   val fields = """class,
   cap-shape, cap-surface, cap-color, bruises, odor,
   gill-attachment, gill-spacing, gill-size, gill-color, stalk-shape,
   stalk-root, stalk-surface-above-ring, stalk-surface-below-ring,
   stalk-color-above-ring, stalk-color-below-ring, veil-type,
   veil-color, ring-number, ring-type, spore-print-color, population, habitat"""
   val flds = fields.split(",")

   val rawData = spark.read.format("csv")
     .option("sep", ",")
     .option("header", "false")
     .load("agaricus-lepiota.data")
     .toDF(fields.split(',').map(_.trim): _*)
  
   // stalk root has 2480 missing entries;
   // hence drop column, not to lose that many rows
   val data = rawData.drop("stalk-root")
  
   data.show(2)

   val indexers = data.columns.map {
     column_name => new StringIndexer()
       .setInputCol(column_name)
       .setOutputCol(column_name + "_index")
   }

   val assembler = new VectorAssembler()
     .setInputCols(data.columns.map(_ + "_index").filterNot(_ contains "class"))
     .setOutputCol("features")

   val transform = new Pipeline().setStages(indexers :+ assembler)

   val transformedData = transform
     .fit(data)
     .transform(data)
     .select("features", "class_index")

   val dl4jDataPrep = transformedData.rdd.map {
     row => Tuple2(
         row.getAs[SparseVector]("features").toDense.toArray,
         row.getAs[Double]("class_index").toInt
       )
   }
  
   val dl4jData = dl4jDataPrep.map {
     elem => new DataSet(
       Nd4j.create(elem._1, Array(1, 21)), // features
       Nd4j.zeros(1, 2).putScalar(elem._2, 1.0) // vector with 1 on correct class' position,
     )
   }.repartition(8)


   val splitted = dl4jData.randomSplit(Array(0.5, 0.5), seed = 5L)
   val trainData = splitted(0)
   val testData = splitted(1)
  
   // we have 21 features
   val inputSize = 21
   // classes are poisonous and edible
   val numClasses = 2



   val conf = new NeuralNetConfiguration.Builder()
     .seed(12345)
     .activation(Activation.RELU)
     .weightInit(WeightInit.XAVIER)
     .updater(new Nesterovs(0.02))
     .l2(1e-4)
     .list()
     .layer(0, new DenseLayer.Builder().nIn(inputSize).nOut(100).build())
     .layer(1, new DenseLayer.Builder().nIn(100).nOut(50).build())
     .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
       .activation(Activation.SOFTMAX).nIn(50).nOut(numClasses).build())
     .build()


     val tm  = new ParameterAveragingTrainingMaster
       .Builder(1)
       .batchSizePerWorker(5000)
       .averagingFrequency(10)
       .workerPrefetchNumBatches(2)
       .build()

    
     val sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

     val numEpochs = 250;

     for (i <- 0 to numEpochs) {
       sparkNet.fit(trainData);
       println(s"Finish $i epoch training");
     }

     val evaluation = sparkNet.doEvaluation(testData, 256, new Evaluation(numClasses))(0)     
     print(evaluation.stats())

     tm.deleteTempFiles(sc);
 }
}
