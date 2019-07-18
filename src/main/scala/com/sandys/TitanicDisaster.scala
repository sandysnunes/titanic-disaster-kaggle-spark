package com.sandys

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Encoders, SQLContext, SparkSession}

/**
 * https://www.kaggle.com/c/titanic/data
 */
object TitanicDisaster extends App {

  val trainingPath = "src/main/resource/train.csv"
  val testPath = "src/main/resource/train.csv"

  val conf: SparkConf = new SparkConf()
    .setAppName("TitanicDisaster")
    .setMaster("local[*]")

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()
  val sc: SparkContext = session.sparkContext
  val sqlContext: SQLContext = session.sqlContext

  import org.apache.spark.sql.functions._
  import sqlContext.implicits._

  val training = ReadCsvWithCache(session, trainingPath)
  val test = ReadCsvWithCache(session, testPath)


  val model: SVMModel = SVMWithSGD.train(training, 100)

  model.clearThreshold()

  val scoreAndLabels = test.map { point =>
    val score = model.predict(point.features)
    (score, point.label)
  }

  // Get evaluation metrics.
  val metrics = new BinaryClassificationMetrics(scoreAndLabels)
  val auROC = metrics.areaUnderROC()

  println(s"Area under ROC = $auROC")

}
