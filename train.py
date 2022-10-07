import sqlite3
import pandas as pd
import pyspark
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

from pyspark.sql.types import StructType,StructField, StringType,FloatType
from pyspark.ml import feature, Pipeline, classification
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder,CrossValidatorModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
from tabulate import tabulate
from pyspark.ml.pipeline import PipelineModel


SQLdatabase_path = "database.sqlite"
model_path       = "lr"
report_path      = "report"
os.mkdir(report_path)
############# Configure Database and Spark Session 

database = sqlite3.connect(SQLdatabase_path)
spark = pyspark.sql.SparkSession.builder.config("spark.driver.memory", "5g").appName("amazonReviews").getOrCreate()

database_cur = database.cursor()
### get table name
database_cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_name = database_cur.fetchall()[0][0]


### get list of cols in table:
data = database_cur.execute("SELECT * FROM {}".format(table_name))
columns = []
for column in data.description:
    columns.append(column[0])
print(columns)

###################### Preprocessing data 


### select desired column, we will only to work with ["HelpfulnessNumerator", "HelpfulnessDenominator", "score", "summary", "text"]
desired_columns = ["HelpfulnessNumerator", "HelpfulnessDenominator", "score", "summary", "text"]

### Filtering Querey, you will find the purpose of every line beside each query line

# Retrieve desired and column and create new column which is the concatenation of summary and text columns
query = "SELECT HelpfulnessNumerator, HelpfulnessDenominator, Score, Summary, Text, Summary || ' and ' || Text as summary_text FROM Reviews  "

# Check type of elements of HelpfulnessDenominator to be real or integer and not NULL
query+= "WHERE (typeof(HelpfulnessDenominator) = 'real' OR typeof(HelpfulnessDenominator) = 'integer') and HelpfulnessDenominator IS NOT NULL "

# Check type of elements of HelpfulnessNumerator to be real or integer and not NULL
query+= "and (typeof(HelpfulnessNumerator) = 'real' OR typeof(HelpfulnessNumerator) = 'integer') and HelpfulnessNumerator IS NOT NULL "

# Check type of elements of Score to be real or integer and not NULL
query+= "and (typeof(Score) = 'real' OR typeof(Score) = 'integer' and HelpfulnessNumerator IS NOT NULL) "  

# Check summary not to be empty or NULL
query+= "and (summary IS NOT NULL) and (summary != '') " 

# Check text not to be empty or NULL
query+= "and (text IS NOT NULL) and (text != '')" 

# Retrieve all columns where HelpfulnessNumerator is greater than or equal HelpfulnessDenominator
query+= "and HelpfulnessNumerator >= HelpfulnessDenominator " 

# Balancing the data by retrieving only 200000
query+= "ORDER BY Score ASC LIMIT 200000" 

# execute query
data = database_cur.execute(query)

###################### SQL to PYSPARK 

### create data frame from retrieved data
Schema = StructType([
  StructField('summary_text', StringType(), True),
  StructField('Score', FloatType(), True),
  ])

df = []
for row in data:
    score = float(row[2])
    summary_text = str(row[-1])
    df.append((summary_text,score))

df = spark.createDataFrame(df, Schema)


###################### Feature Engineering 
tokenizer         = feature.Tokenizer(inputCol='summary_text',outputCol='summary_text_tokens')
stopwords_remover = feature.StopWordsRemover(inputCol='summary_text_tokens',outputCol='filtered_tokens')
vectorizer        = feature.CountVectorizer(inputCol='filtered_tokens',outputCol='features')
idf               = feature.IDF(inputCol='features',outputCol='vectorized_features')



###################### Split Data       
(train_df,test_df) = df.randomSplit((0.75,0.25),seed=42)


###################### Create Model
estimator = classification.LogisticRegression(maxIter = 10,featuresCol='vectorized_features',labelCol='Score')
pipeline  = Pipeline(stages=[tokenizer,stopwords_remover,vectorizer,idf,estimator])

paramGrid_lr = ParamGridBuilder() \
    .addGrid(estimator.regParam, np.linspace(0.3, 0.01, 5)) \
    .build()

crossval_lr  = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid_lr,
                          evaluator=MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='Score'),
                          numFolds= 5) 



######################   Training
lr_model    = crossval_lr.fit(train_df)
lr_model.save(model_path)


######################   Evaluate
predictions = lr_model.transform(test_df)

preds_and_labels = predictions.select(['prediction','Score']).withColumn('label', col('Score')).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

### confusion_matrix
confusion_matrix = metrics.confusionMatrix().toArray()
ax = sns.heatmap(confusion_matrix/np.sum(confusion_matrix,axis=1).reshape((confusion_matrix.shape[0],1)), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Confusion Matrix\n')
ax.set_xlabel('Predicted Score')
ax.set_ylabel('Actual Score ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['1', '2', '3', '4', '5'])
ax.yaxis.set_ticklabels(['1', '2', '3', '4', '5'])

fig = ax.get_figure()
fig.savefig(os.path.join(report_path,"confusion_matrix.png"))

with open(os.path.join(report_path,'report.txt','w')) as w:
    table   = []
    headers = ["Metric", "Result"]
    table.append(["Accuracy", str(np.round(metrics.accuracy,3))])
    table.append(["Weighted Precision", str(np.round(metrics.weightedPrecision,3))])
    table.append(["Weighted Recall", str(np.round(metrics.weightedRecall,3))])
    table.append(["Weighted F1", str(np.round(metrics.weightedFMeasure(),3))])
    w.write(tabulate(table, headers=headers))

    w.write('\n\nMetrics Per Label:\n')
    table   = []
    headers = ["Label","Precision", "Recall","F1"]
    for label in [1.0,2.0,3.0,4.0,5.0]:
        table.append([str(label),str(np.round(metrics.precision(label),3)),str(np.round(metrics.recall(label),3)),str(np.round(metrics.fMeasure(label),3))])

    w.write(tabulate(table, headers=headers))

database.close()
