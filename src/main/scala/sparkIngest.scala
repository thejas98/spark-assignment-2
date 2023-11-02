import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._


object sparkIngest extends App{



  // 1. Create a Spark session
  val spark = SparkSession.builder()
    .appName("TitanicSurvivalPrediction")
    .master("local")
    .getOrCreate()
  // 2. Load the data
  val trainDF = spark.read.option("header", "true").option("inferSchema", "true").csv("/home/thejas/projects/spark-assignment-2/data/titanic/train.csv")
  val testDF = spark.read.option("header", "true").option("inferSchema", "true").csv("/home/thejas/projects/spark-assignment-2/data/titanic/test.csv")

  // 3. Exploratory Data Analysis (EDA)
  trainDF.show()
  trainDF.printSchema()


  // Mean median and other
  trainDF.select().summary().show()

  // Count of missing values
  trainDF.select(trainDF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()

  // Categorical variables
  trainDF.groupBy("Sex").count().show()
  trainDF.groupBy("Embarked").count().show()

  // Correlation
  trainDF.stat.corr("Age", "Fare")

  // Survival rate by gender
  trainDF.groupBy("Sex").agg(avg("Survived")).show()

  // Age distribution by class
  trainDF.groupBy("Pclass").agg(avg("Age")).show()

  // Survival rate by port of embarkation
  trainDF.groupBy("Embarked").agg(avg("Survived")).show()

  // 4. Handle the null values in the dataset

  // the Age, Fare, Cabin, and Embarked columns have null values
  trainDF.summary("count").show
  testDF.summary("count").show

  val avgAge = trainDF.select("Age").unionAll(testDF.select("Age"))
    .agg(avg("Age"))
    .collect() match {
    case Array(Row(avg: Double)) => avg
    case _ => 0
  }

  val avgFare = trainDF.select("Fare").union(testDF.select("Fare"))
    .agg(avg("Fare"))
    .collect() match {
    case Array(Row(avg: Double)) => avg
    case _ => 0
  }


  val filledDf_train = trainDF.na.fill(Map("Fare" -> avgFare, "Age" -> avgAge, "Embarked" -> "S"))

  val filledDf_test = testDF.na.fill(Map("Fare" -> avgFare, "Age" -> avgAge, "Embarked" -> "S"))

  filledDf_train.summary().show()
  filledDf_test.summary().show()



  // 5. Add features to the dataset
  val trainDF_transformed = filledDf_train
    .withColumn("IsAlone", when(trainDF("SibSp") + trainDF("Parch") === 0, 1).otherwise(0))
    .withColumn("FamilySize", trainDF("SibSp") + trainDF("Parch"))

  val testDF_transformed = filledDf_test
    .withColumn("IsAlone", when(testDF("SibSp") + testDF("Parch") === 0, 1).otherwise(0))
    .withColumn("FamilySize", testDF("SibSp") + testDF("Parch"))
    .withColumn("Survived", lit("0")).select("PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","IsAlone","FamilySize")


  //Prepare ML MODEl data
  val trainDFFinal = trainDF_transformed.drop("PassengerId", "Name", "Ticket", "Cabin")
  val testDFFinal = testDF_transformed.drop("PassengerId", "Name", "Ticket", "Cabin")

  val allData = trainDFFinal.union(testDFFinal)
  allData.cache()








  // 5. Model Training

  val featureCols = Seq("SibSp", "Parch", "Fare", "Age", "FamilySize", "IsAlone")
  val categoryFeatures = Seq("Pclass", "Sex", "Embarked")

  val stringIndexers = categoryFeatures.map { colName =>
    new StringIndexer()
      .setInputCol(colName)
      .setOutputCol(colName + "Indexed")
      .fit(trainDFFinal)
  }

  //Indexing target feature
  val labelIndexer = new StringIndexer()
    .setInputCol("Survived")
    .setOutputCol("SurvivedIndexed")
    .fit(trainDFFinal)


  val categoryIndexedFeatures = categoryFeatures.map(_ + "Indexed")

  val IndexedFeatures = featureCols ++ categoryIndexedFeatures


  val assembler = new VectorAssembler()
    .setInputCols(Array(IndexedFeatures: _*))
    .setOutputCol("features")

  val randomForest = new RandomForestClassifier()
    .setLabelCol("Survived")
    .setFeaturesCol("features")


  //Retrieving original labels
  val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

  val pipeline = new Pipeline().setStages((stringIndexers :+ labelIndexer :+ assembler :+ randomForest :+ labelConverter).toArray)


  val evaluator = new MulticlassClassificationEvaluator().setLabelCol("SurvivedIndexed").setPredictionCol("prediction")

  val model = pipeline.fit(trainDFFinal)
  val predictions = model.transform(testDFFinal)

  val accuracy = evaluator.evaluate(predictions)

  println("Accuracy of Titanic Train and Test: " + accuracy * 100)


}
