package com.sandys

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Encoders, SQLContext, SparkSession}

/**
 * https://www.kaggle.com/c/titanic/data
 */
object TitanicDisaster extends App {

  val trainingPath = "src/main/resource/train.csv"
  val testPath = "src/main/resource/test.csv"

  val conf: SparkConf = new SparkConf().setAppName("TitanicDisaster").setMaster("local[*]")

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()

  val training = ReadCsvWithCache(session, trainingPath)
  val test = ReadCsvWithCache(session, testPath)


  val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)

  // Compute raw scores on the test set.
  val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
    val prediction = model.predict(features)
    (prediction, label)
  }

  // Get evaluation metrics.
  val metrics = new MulticlassMetrics(predictionAndLabels)
  println(s"Accuracy = ${metrics.accuracy}")


}
