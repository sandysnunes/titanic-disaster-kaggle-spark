package com.sandysnunes

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}

object ReadCsvWithCache {

  private val PassengerId = "PassengerId"
  private val Survived = "Survived"
  private val Pclass = "Pclass"
  private val Name = "Name"
  private val Sex = "Sex"
  private val Age = "Age"
  private val SibSp = "SibSp"
  private val Parch = "Parch"
  private val Ticket = "Ticket"
  private val Fare = "Fare"
  private val Cabin = "Cabin"
  private val Embarked = "Embarked"

  def apply(spark: SparkSession, path: String): RDD[LabeledPoint] = {

    val dataFrame = spark.read
      .option("header", "true")
      .csv(path)

    dataFrame.rdd
      .filter(row => row.getAs(Age) != null)
      .filter(row => row.getAs(Embarked) != null)
      .filter(row => row.getAs(Fare) != null)
      .map(row => LabeledPoint(row.getAs[String](Survived).toDouble, contructFeaturesVector(row)))
      .cache()
  }

  private def contructFeaturesVector(row: Row): linalg.Vector = {

    Vectors.dense(
      row.getAs[String](Pclass).toDouble,
      row.getAs[String](Sex) match {
        case "female" => 0.0
        case "male" => 1.0
      },
      row.getAs[String](Age).toDouble,
      row.getAs[String](SibSp).toDouble,
      row.getAs[String](Parch).toDouble,
      row.getAs[String](Fare).toDouble,
      row.getAs[String](Embarked) match {
        case "C" => 0.0
        case "S" => 1.0
        case "Q" => 2.0
      }
    )

  }
}


