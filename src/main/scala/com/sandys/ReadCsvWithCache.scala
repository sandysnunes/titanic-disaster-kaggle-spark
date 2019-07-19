package com.sandys

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Encoders, SparkSession}

object ReadCsvWithCache {

  private val PassengerId = "PassengerId"
  private val Survived = "Survived"
  //private val SurvivedIndex = Survived + "Index"
  private val Pclass = "Pclass"
  //private val PclassIndex = Pclass + "Index"
  private val Name = "Name"
  private val Sex = "Sex"
  private val Age = "Age"
  private val SibSp = "SibSp"
  private val Parch = "Parch"
  private val Ticket = "Ticket"
  private val Fare = "Fare"
  private val Cabin = "Cabin"
  private val Embarked = "Embarked"

  private val categoricalColumns  = Seq(Survived, Pclass, Sex, Age, Embarked)

  private val fieldNames = Seq(PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)

  def apply(spark: SparkSession, path: String): RDD[LabeledPoint] = {
    spark.read
      .option("header", "true")
      .csv(path)
      .rdd
      .map(row => {
        val valueByField = row.getValuesMap(fieldNames)

        val _class = valueByField(Survived).asInstanceOf[String].toDouble
        val vectorFeatures = contructVector(valueByField)

        LabeledPoint(_class, vectorFeatures)
      })
      .cache()
  }


  private def contructVector(valueByField: Map[String, Nothing]): linalg.Vector = {
    Vectors.dense(
      valueByField(Pclass).asInstanceOf[String].toDouble,
      valueByField(Sex).asInstanceOf[String] match {
        case "female" => 0.0
        case "male" => 1.0
      },
      valueByField(Age).asInstanceOf[String].toDouble,
      valueByField(SibSp).asInstanceOf[String].toDouble,
      valueByField(Parch).asInstanceOf[String].toDouble,
      valueByField(Fare).asInstanceOf[String].toDouble,
      //valueByField(Cabin).asInstanceOf[String].toDouble
      valueByField(Embarked).asInstanceOf[String] match {
        case "C" => 0.0
        case "S" => 1.0
        case "Q" => 2.0
      }
    )
  }
}
