import pyspark
import sqlite3

import sys

from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder,CrossValidatorModel



SQLdatabase_path = "database.sqlite"
model_path       = "lr"
output_csv_path  = "output.csv"

database = sqlite3.connect(SQLdatabase_path)
spark = pyspark.sql.SparkSession.builder.config("spark.driver.memory", "5g").appName("amazonReviews").getOrCreate()

database_cur = database.cursor()

### retrieve data from database
query = "SELECT ID, Summary || ' and ' || Text as summary_text FROM Reviews"
data = database_cur.execute(query)

### create data frame from retrieved data
Schema = StructType([StructField('ID', IntegerType(), True) ,StructField('summary_text', StringType(), True)])
df = []
for row in data:
    ID = row[0]
    summary_text = str(row[1])
    df.append((ID,summary_text))
df = spark.createDataFrame(df, Schema)

### load model
lr_model = CrossValidatorModel.load(model_path)


predictions = lr_model.transform(df)
predictions = predictions.select('ID','prediction')
predictions.toPandas().to_csv(output_csv_path)

