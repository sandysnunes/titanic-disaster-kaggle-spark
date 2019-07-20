package com.sandys

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

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

  //----------------------------------------------------------------------
  // Train a NaiveBayes model.
  val nvModel: NaiveBayesModel = new NaiveBayes().run(training)

  val nVpredictionAndLabels = test.map { case LabeledPoint(label, features) =>
    val prediction = nvModel.predict(features)
    (prediction, label)
  }

  // Get evaluation metrics.
  val nvMetrics = new MulticlassMetrics(nVpredictionAndLabels)
  println(s"Accuracy = ${nvMetrics.accuracy}")

}
