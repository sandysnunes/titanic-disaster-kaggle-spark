package com.sandysnunes

import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

/**
 * https://www.kaggle.com/c/titanic/data
 */
object TitanicDisaster extends App {

  val conf: SparkConf = new SparkConf().setAppName("TitanicDisaster").setMaster("local[*]")
  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()

  //leitura e pré-processamento dos dados
  val training = ReadCsvWithCache(session, "src/main/resource/train.csv")
  val test = ReadCsvWithCache(session, "src/main/resource/test.csv")

  val logisticRegression = new LogisticRegressionWithLBFGS().setNumClasses(2)
  val model = logisticRegression.run(training)

  // Computa a classificação e junta com a classe
  val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
    val prediction = model.predict(features)
    (prediction, label)
  }

  // Calcula as métricas
  val metrics = new MulticlassMetrics(predictionAndLabels)
  println(s"Accuracy = ${metrics.accuracy}")

}
