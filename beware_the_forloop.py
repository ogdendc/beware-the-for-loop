# Databricks notebook source
# MAGIC %md
# MAGIC # Comparing *sequential* vs *parallel* processing

# COMMAND ----------

# MAGIC %md ## The premise of the exercise below:
# MAGIC > #### Within our data there are *many* "groups". 
# MAGIC >> ##### A common example of high-cardinality groups would be microgeographies such as zipcodes. 
# MAGIC > #### We want to calculate some metrics or perform some functions (or even modeling) on each group separately.
# MAGIC > #### We might be a bit Spark-timid, or perhaps tend towards non-distributed tools such as Pandas.
# MAGIC > #### We are using a for-loop to curate data, and are surprised by how long it runs.

# COMMAND ----------

# MAGIC %md ## Getting started:
# MAGIC > #### This notebook generates its own data to be used.
# MAGIC > #### To see similar run times, create/use a cluster equivalent (or similar) to the configuration shown below.
# MAGIC > #### Total expected end-to-end run time of this notebook is ~ 45 minutes.
# MAGIC > #### By design, a majority of that time is from running a couple of for-loops.
# MAGIC > #### Suggested cluster specs below:

# COMMAND ----------

# DBTITLE 1,Azure:
# MAGIC %md
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=1WnImMAjT89pQ13MRVOtWpAyRr8irTuFF' width="700" height="1100">

# COMMAND ----------

# DBTITLE 1,measuring start time of the notebook
import datetime
nb_start = datetime.datetime.now() # measuring total run time of the notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ### our synthetic data:
# MAGIC >#### summary of units sold
# MAGIC >#### by location_ID, microsegment_ID, and transaction_date
# MAGIC >#### notice that there is discontinuity in the dates

# COMMAND ----------

# MAGIC %md #### Side note:  We are indeed using a for-loop to generate our synthetic data.  
# MAGIC > #### We'll rethink this later, below, when we stress-test our approach.

# COMMAND ----------

# DBTITLE 1,defining dimensionality of the synthetic data
import pandas as pd

dr = pd.date_range(start='1/1/2022', end='12/31/2022') # relevant continuous dates for data history
L = 1000                                               # number of locations, our "group by" variable
M = 100                                                # number of microsegments, for some granularity in our summarized history

# COMMAND ----------

# DBTITLE 1,using a for-loop to generate synthetic data
import random

output = []  # <-- our results will be stored as a list, then converted to a Spark dataframe

for i in range(L):        # L = number of locations defined above
  for j in range(M):        # M = number of microsegments defined above
    for t in dr:              # dr = list of relevant (continuous) dates defined above
      # random integer from -1 to 10
      n = random.randint(-1, 10)     # used to randomly delete rows from our granular data, mimicking operational irregularity
      if n != -1 & t.weekday() < 6:  # forcing some granular discontinuity in the dates & assuming all locations are closed on Sundays
        output.append((i+1000, j+1, t.date(), n))     
        
spark_df = spark.createDataFrame(output).toDF("location_ID", "microsegment_ID", "transaction_date", "units_sold")
display(spark_df)

# COMMAND ----------

# DBTITLE 1,observing the size of the synthetic data
# getting shape for spark dataframe = rows & columns
print(("table size in rows, columns: ", spark_df.count(), len(spark_df.columns)))

# COMMAND ----------

# MAGIC %md ## Notes about the synthetic data, above:
# MAGIC > #### With suggested cluster config, it takes ~ 13 minutes to generate our synthetic data...
# MAGIC > #### ...using a for-loop approach.
# MAGIC > #### With 1000 "locations" and 100 "microsegments" and 365 days...
# MAGIC > #### ...and with some removal of some dates...
# MAGIC > #### ...we end up with 28+ million rows.

# COMMAND ----------

# MAGIC %md ## Partitioning the underlying data by the groupBy column:

# COMMAND ----------

# DBTITLE 1,checking starting partitions out of curiosity
print("beginning partitions = ", spark_df.rdd.getNumPartitions())

# COMMAND ----------

# DBTITLE 1,partitioning by our "groups" variable will significantly reduce run times below
# partitioning the underlying data by the group-by column
spark_df=spark_df.repartition('location_ID')

print("new # of partitions = ", spark_df.rdd.getNumPartitions())

# Note:  For a truly BIG DATA scenario, repartitioning could be a prohibitively expensive and time consuming operation.
#        Some testing should be done to determine if this step is warranted for your particular situation.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Our goal:
# MAGIC - #### summarize units sold by location_ID & transaction_date (i.e. without the 'microsegment' detail)
# MAGIC - #### filling-in the discontinuity of dates (assume zero units_sold for missing dates)

# COMMAND ----------

# DBTITLE 1,unique combinations for group-by processing, saved as a Pandas dataframe
# distinct values of location_ID, used in subsequent steps
group_list = spark_df.select('location_ID').distinct().sort('location_ID').toPandas()  
display(group_list)

# COMMAND ----------

# DBTITLE 1,confirming (rows, columns) for the list of unique "groups"
# checking number of unique groups (i.e. unique location_IDs) 
group_list.shape

# COMMAND ----------

# MAGIC %md # Method 1:  the sequential for-loop approach:

# COMMAND ----------

# MAGIC %md > #### output will be a dictionary of Pandas dataframes

# COMMAND ----------

# DBTITLE 1,measuring start time for this method
start = datetime.datetime.now()

# COMMAND ----------

df_dict = {}   # <-- empty dictionary to hold results from iterative for-loop

# same list of relevant dates created above
#dr = pd.date_range(start='1/1/2022', end='12/31/2022')  # <-- used below to fill-in gaps in dates in the source data, via Pandas re-index

for i in group_list.location_ID:  # <-- looping through each location_ID
  # for each iteration, select the relevant records and convert subset to a Pandas dataframe
  pandas_temp_df = spark_df.filter((spark_df.location_ID == i)).toPandas()
  # populate entry in the dictionary of dataframes as units_sold summed by transaction_date (for given location_ID)
  df_dict[i] = pandas_temp_df.groupby(["transaction_date"])["units_sold"].sum().reset_index()
  # use our Pandas continuous date range ("dr") to fill-in gaps in the data/dates
  df_dict[i] = df_dict[i].set_index('transaction_date').reindex(dr).fillna(0).rename_axis('transaction_date').reset_index()

# COMMAND ----------

# DBTITLE 1,measure approximate run time for the method above
end = datetime.datetime.now()
elapsed = end - start
minutes = round(elapsed.total_seconds() / 60, 3)
print("approximate run time for this method = ", minutes, " minutes")

# COMMAND ----------

# MAGIC %md > #### the keys in the dictionary are the distinct groups per the for-loop iterations
# MAGIC > #### and the values in the dictionary are the summary Pandas dataframes

# COMMAND ----------

display(df_dict)

# COMMAND ----------

# MAGIC %md > #### this optional step collapses the above dictionary into a single dataframe:

# COMMAND ----------

# concat individual dfs into a single output dataframe
grouped_pandas_df = pd.concat(df_dict, axis=0).reset_index(level=0).rename({'level_0':'location_ID'}, axis=1)

# converting resulting datetime to just date
grouped_pandas_df['transaction_date'] = pd.to_datetime(grouped_pandas_df['transaction_date']).dt.date

display(grouped_pandas_df)

# COMMAND ----------

# DBTITLE 1,validating the size of the output (# of groups times # of days in the year)
# shape of the output in rows, columns
grouped_pandas_df.shape                  # <-- should be # of groups x 365 (dates)

# COMMAND ----------

# MAGIC %md # Method 2:  the groupBy.applyinPandas (parallel) approach:

# COMMAND ----------

# Step 1:  start with a Spark (not Pandas) dataframe

# (same one as initially read-in / created above)

# COMMAND ----------

# Step 2:  create a Pandas user-defined-function (udf) that performs desired calcs for each group

# Note:  the applyinPandas API takes a Spark dataframe as input, but...
#        we write our udf for Pandas functions, as the API converts each data subset into a Pandas dataframe...
#        and the API collects/appends the results and returns a single Spark dataframe.

# Notice:
#        the steps in the udf below are essentially the same as those in the for-loop above, however...
#        the incoming subset is already in Pandas, and...
#        the output is a Spark dataframe and thus (unlike above) *not* a dictionary of Pandas dataframes, so...
#        there are a few required minor coding tweaks to what is seen in the for-loop above.

def my_udf(input_data):
  current_group = input_data['location_ID'].iloc[0]
  input_data = input_data.groupby(["location_ID","transaction_date"])["units_sold"].sum().reset_index()
  input_data = input_data.set_index('transaction_date').reindex(dr).fillna({'location_ID':current_group,'units_sold':0}).rename_axis('transaction_date').reset_index()
  return input_data 

# COMMAND ----------

# Step 3:  create schema for the Spark dataframe to be returned by the applyinPandas API

from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType, IntegerType, ArrayType, MapType, DecimalType, DateType, TimestampType

#sSpecify Spark dataframe schema
spark_schema = StructType([
               StructField('location_ID',               IntegerType(),  True),
               StructField('transaction_date',          DateType(),    True),
               StructField('units_sold',                IntegerType(),  True)          
               ])

# Side note: 
#        Needing to define the schema of the returned output does at first seem like a bit of tedium.  However...
#        ...when we see the incredible improvement in processing time...
#        ...this technical overhead is a small price to pay!

# COMMAND ----------

# DBTITLE 1,measuring start time for this method
start = datetime.datetime.now()

# COMMAND ----------

# Step 4:  do the Spark groupBy function with applyinPandas API

# Note:  the .sort of the output does add unnecessary processing time, but provides more intuitively displayed output

grouped_spark_df = spark_df.groupBy('location_ID').applyInPandas(my_udf, schema=spark_schema).sort('location_ID','transaction_date') 
display(grouped_spark_df)  # <-- this steps actually executes the API call, thanks to the lazy evaluation of Spark

# COMMAND ----------

# DBTITLE 1,measuring approximate run time for the above method
end = datetime.datetime.now()
elapsed = end - start
minutes = round(elapsed.total_seconds() / 60, 3)
print("approximate run time for this method = ", minutes, " minutes")

# COMMAND ----------

# MAGIC %md ## Let's pause and reflect...
# MAGIC > #### on how long it took Method 1 (for-loop) vs Method 2 (gropuBy.applyinPandas):
# MAGIC - #### for-loop ~ 15 minutes
# MAGIC - #### groupBy.applyinPandas ~ 6 *seconds*

# COMMAND ----------

# MAGIC %md # Note on run times:
# MAGIC > ### Results will vary.
# MAGIC > ##### You may not get the same run times, but...
# MAGIC > ##### you should see the same order of magnitude of difference between run times for the various methods.

# COMMAND ----------

# DBTITLE 1,validating the size of the output (# of groups times # of days in the year)
# shape of the output in rows, columns
print(("table size in rows, columns: ", grouped_spark_df.count(), len(grouped_spark_df.columns)))

# COMMAND ----------

# MAGIC %md > #### an optional step to convert to a dictionary of Pandas dataframes...
# MAGIC > #### if that happened to be the desired format of the output

# COMMAND ----------

grouped_pandas_df = grouped_spark_df.toPandas()  # <-- converting from Spark to Pandas dataframe

# creating dictionary of Pandas dataframes from the single Pandas dataframe
df_dict = {}
for i in group_list.location_ID:
  df_dict[i]=grouped_pandas_df.loc[grouped_pandas_df["location_ID"]==i].set_index('transaction_date').reset_index().drop(["location_ID"], axis=1)
  
display(df_dict)

# COMMAND ----------

# MAGIC %md # Method 3:  the multi-threaded approach:

# COMMAND ----------

# MAGIC %md > #### somewhere between the sequential for-loop and the parallel applyinPandas...
# MAGIC > #### we can force some parallelization using the concurrent.futures module with the ThreadPoolExecutor subclass.
# MAGIC > #### i.e. force a multi-threaded process onto our Python / Pandas (non-Spark) executable 

# COMMAND ----------

# writing a udf that defines what is to be performed by each thread
#      very similar to the udf above, with some minor coding tweaks

def my_new_udf(x):
  grouped_pandas_df = spark_df.filter((spark_df.location_ID == x)).toPandas()
  grouped_pandas_df = grouped_pandas_df.groupby(["location_ID","transaction_date"])["units_sold"].sum().reset_index()
  grouped_pandas_df = grouped_pandas_df.set_index('transaction_date').reindex(dr).fillna({'location_ID':x,'units_sold':0}).rename_axis('transaction_date').reset_index()
  return grouped_pandas_df  

# COMMAND ----------

# MAGIC %md ##### Note: the ThreadPool approach appears to be negatively impacted by too many partitions.
# MAGIC > ###### repartitioning by the groupby column works fine, as long as you don't force more than default partitions.

# COMMAND ----------

# DBTITLE 1,measuring start time for this method
start = datetime.datetime.now()

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor

results = [] # <-- our results will be stored as a List of Pandas dataframes

with ThreadPoolExecutor(max_workers=1000) as threadpool:
  results.extend(threadpool.map(lambda x: my_new_udf(x), group_list.location_ID))
  
# Notice:
#       If we look at the Ganglia report during this step...
#       ...we'd see a significant increase in CPU utilization on the workers...
#       ...which is what we'd expect/want for this forced multithreading.

# COMMAND ----------

# DBTITLE 1,measuring approximate run time for the method above
end = datetime.datetime.now()
elapsed = end - start
minutes = round(elapsed.total_seconds() / 60, 3)
print("approximate run time for this method = ", minutes, " minutes")

# COMMAND ----------

# MAGIC %md ## Let's pause and reflect...
# MAGIC > #### on how long it took Method 3 (ThreadPool) vs the other methods above:
# MAGIC - #### for-loop ~ 15 minutes
# MAGIC - #### groupBy.applyinPandas ~ 6 *seconds*
# MAGIC - #### ThreadPool ~ 7 minutes

# COMMAND ----------

# MAGIC %md ## Note:  If we observed cluster utilization (see Ganglia reports) during this execution...
# MAGIC > #### we'd see significant increase in CPU utilization during ThreadPool execution...
# MAGIC > #### which is what we expect/want from this approach.

# COMMAND ----------

# looking at the resulting *list* of Pandas df's
results

# COMMAND ----------

# MAGIC %md > #### this optional step collapses the above *list* into a single Pandas dataframe:

# COMMAND ----------

# concat individual dfs into a single output dataframe
grouped_pandas_df = pd.concat(results, axis=0, ignore_index=True)
# converting resulting datetime to just date
grouped_pandas_df['transaction_date'] = pd.to_datetime(grouped_pandas_df['transaction_date']).dt.date
display(grouped_pandas_df)

# COMMAND ----------

# DBTITLE 1,validating the size of the output (# of groups times # of days in the year)
# shape of the output in rows, columns
grouped_pandas_df.shape

# COMMAND ----------

# MAGIC %md > ##### above, we saw how easily we can convert to a dictionary of Pandas dataframes...
# MAGIC > ###### from the single Pandas dataframe
# MAGIC > ###### if that happened to be the desired format of the output

# COMMAND ----------

# MAGIC %md # Method 4:  a PySpark (no Pandas) groupBy approach:

# COMMAND ----------

# MAGIC %md > #### What if we are open to, and able to, accomplish desired results *without* Pandas?
# MAGIC > #### The premise of above 3 methods assumes we want or need to use Pandas functionality.
# MAGIC > #### If we can (and are willing to) rethink our code, and replicate Pandas functionality with just PySpark...

# COMMAND ----------

# DBTITLE 1,measuring start time for this method
start = datetime.datetime.now()

# COMMAND ----------

from pyspark.sql.functions import lit, col, when, sum as _sum # <-- default sum in the agg function returns an error for no good reason

# summarizing units_sold by location and date...but...
#      the output will have gaps in the dates, which we used Pandas reindex to address above
grouped_spark_df = spark_df.groupBy('location_ID', 'transaction_date').agg(_sum('units_sold').alias('units_sold'))

# to address the missing dates without Pandas...
# one option is to create an "exploded" dataframe having all locations with all (continuous, relevant) dates:
id_dates_df = grouped_spark_df.selectExpr("location_ID").distinct().selectExpr("location_ID","explode(sequence(date('2022-01-01'),date('2022-12-31'),INTERVAL 1 DAY)) as transaction_date") 

# then we do a left join of the summary units_sold onto the "exploded" dataframe having all dates...
grouped_spark_df = id_dates_df.alias("a").join(grouped_spark_df.alias("b"), (col("a.location_ID") == col("b.location_ID")) & (col("a.transaction_date") == col("b.transaction_date")),'left').select(col("a.location_ID"), col("a.transaction_date"), col("b.units_sold"))
# Note:  In above line, using the dataframe.alias approach to avoid this issue:  https://issues.apache.org/jira/browse/SPARK-14948
#          Tried many solutions, including renaming id columns, but using the .alias is what worked.
#          Original line of code that caused the ambiguity issue is shown below:
#grouped_spark_df = id_dates_df.join(grouped_spark_df, (id_dates_df.location_ID == grouped_spark_df.location_ID) & (id_dates_df.transaction_date == grouped_spark_df.transaction_date),'left').select(id_dates_df.location_ID, id_dates_df.transaction_date, grouped_spark_df.units_sold)

# with a step to replace resulting NULL units_sold with a value of zero:
grouped_spark_df = grouped_spark_df.withColumn('units_sold', when(col('units_sold').isNull(), 0).otherwise(col('units_sold'))).sort('location_ID','transaction_date') 

display(grouped_spark_df)

# COMMAND ----------

# DBTITLE 1,measuring approximate run time for the method above
end = datetime.datetime.now()
elapsed = end - start
minutes = round(elapsed.total_seconds() / 60, 3)
print("approximate run time for this method = ", minutes, " minutes")

# COMMAND ----------

# DBTITLE 1,validating the size of the output (# of groups times # of days in the year)
# shape of the output in rows, columns
print(("table size in rows, columns: ", grouped_spark_df.count(), len(grouped_spark_df.columns)))

# COMMAND ----------

# MAGIC %md ### A note about the above PySpark approach:
# MAGIC > #### If the calcs we needed to do were more complex than a simple sum...
# MAGIC > #### we could write a *udf* using only PySpark code, and rather than groupBy.agg
# MAGIC > #### we could do a groupBy.apply(udf) approach.
# MAGIC > #### This would be similar to the groupBy.applyinPandas(udf), sans Pandas.

# COMMAND ----------

# MAGIC %md ##### As we've seen above...
# MAGIC ###### this Spark dataframe could easily be converted to a Pandas dataframe...
# MAGIC ###### and/or dictionary of Pandas dataframes...
# MAGIC ###### if that's the desired format for downstream purposes.

# COMMAND ----------

# MAGIC %md ## Reviewing above run times:
# MAGIC - ### for-loop ~ 15 minutes
# MAGIC - ### groupBy.applyinPandas ~ 6 *seconds*
# MAGIC - ### ThreadPool ~ 7 minutes
# MAGIC - ### PySpark (no Pandas) ~ 5 *seconds*

# COMMAND ----------

# MAGIC %md #Next: stress-testing the faster methods.
# MAGIC >### Running the groupBy.applyinPandas and the PySpark-only approaches on a larger set of data:

# COMMAND ----------

# MAGIC %md ## First, let's revisit how we created the synthetic data.
# MAGIC > #### Using a for-loop (above) ran for > 13 minutes.
# MAGIC > #### As seen above, *beware the for-loop*!

# COMMAND ----------

# MAGIC %md ## Creating synthetic data using PySpark (no for-loop!)

# COMMAND ----------

# DBTITLE 1,producing synthetic data with a much faster approach than a for-loop
from pyspark.sql.functions import rand, round as _round, dayofweek

# dataframe with all distinct locations
df1 = spark.range(1000,1000+L).withColumnRenamed("id","location_ID") # L location_ID values from 1000 to 1000+L-1
# dataframe with all distinct microsegments
df2 = spark.range(1,1+M).withColumnRenamed("id","microsegment_ID") # M microsegment_ID values from 1 to M
# crossjoin locations*microsegments * explode of all dates in relevant range
spark_df = df1.crossJoin(df2).selectExpr("location_ID","microsegment_ID","explode(sequence(date('2022-01-01'),date('2022-12-31'),INTERVAL 1 DAY)) as transaction_date").withColumn("units_sold", _round(rand()*(10.4999--1.4999)-1.4999,0)).withColumn("dayofweek", dayofweek('transaction_date'))
# Note:  above round(rand()*(10.4999--1.4999)-1.4999,0) yields a PySpark-based uniform random integer
#          used to randomly delete rows from our granular data (as was done in the for-loop approach)
spark_df = spark_df.filter((spark_df.dayofweek > 1) & (spark_df.units_sold >= 0)).drop('dayofweek').sort('location_ID','microsegment_ID','transaction_date')

display(spark_df)

# COMMAND ----------

# checking the size of the resulting generated synthetic data
print(("table size in rows, columns: ", spark_df.count(), len(spark_df.columns)))

# COMMAND ----------

# MAGIC %md ## Note:  Due to using a random number generator as part of our logic...
# MAGIC - #### this synthetic data generated without a for-loop (PySpark, no Pandas)...
# MAGIC >> #### won't be the *exact* same data as generated using a for-loop.
# MAGIC - #### But is effectively the same in terms of structure, content, and size.
# MAGIC - ### This no-for-loop approach ran in ~ 3 *seconds* as compared to ~ 13 minutes!
# MAGIC ## Thus, we will *not* use a for-loop to create a larger set of synthetic data:

# COMMAND ----------

# MAGIC %md ## Creating LARGER synthetic data using PySpark (no for-loop!)

# COMMAND ----------

# DBTITLE 1,new data will have ~ 100x the rows and 10x the number of "groups"
L = 10000   # 10x that of above
M = 1000    # 10x that of above

# COMMAND ----------

# DBTITLE 1,Same no-for-loop method shown above, with more locations & more microsegments:
df1 = spark.range(1000,1000+L).withColumnRenamed("id","location_ID") # L location_ID values from 1000 to 1000+L-1
df2 = spark.range(1,1+M).withColumnRenamed("id","microsegment_ID") # M microsegment_ID values from 1 to M
spark_df = df1.crossJoin(df2).selectExpr("location_ID","microsegment_ID","explode(sequence(date('2022-01-01'),date('2022-12-31'),INTERVAL 1 DAY)) as transaction_date").withColumn("units_sold", _round(rand()*(10.4999--1.4999)-1.4999,0)).withColumn("dayofweek", dayofweek('transaction_date'))
spark_df = spark_df.filter((spark_df.dayofweek > 1) & (spark_df.units_sold >= 0)).drop('dayofweek').sort('location_ID','microsegment_ID','transaction_date')

display(spark_df)

# COMMAND ----------

# DBTITLE 1,observing size of our larger synthetic data
# checking the size of the resulting generated synthetic data
print(("table size in rows, columns: ", spark_df.count(), len(spark_df.columns)))

# COMMAND ----------

# MAGIC %md #### As we did above, repartioning the data based on the groupBy column does improve execution time:

# COMMAND ----------

# DBTITLE 1,increasing max shuffle partitions to allow for > default, as our data volume & # of groups grows
sqlContext.setConf("spark.sql.shuffle.partitions", "2001")  #<-- as our # of groups increases, increasing this ceiling

# COMMAND ----------

# DBTITLE 1,partitioning the data by our "groups", just as done with the smaller data above
# partitioning the underlying data by the group-by column
spark_df=spark_df.repartition('location_ID')

print("new # of partitions = ", spark_df.rdd.getNumPartitions())

# Note:  For a truly BIG DATA scenario, repartitioning could be a prohibitively expensive and time consuming operation.
#        Some testing should be done to determine if this step is warranted for your particular situation.

# COMMAND ----------

# MAGIC %md ## groupBy.applyinPandas approach on the larger data:

# COMMAND ----------

# DBTITLE 1,measuring start time for this method
start = datetime.datetime.now()

# COMMAND ----------

# Step 4:  do the Spark groupBy function with applyinPandas API

# Note:  the .sort of the output does add unnecessary processing time, but provides more intuitively displayed output

grouped_spark_df = spark_df.groupBy('location_ID').applyInPandas(my_udf, schema=spark_schema).sort('location_ID','transaction_date') 
display(grouped_spark_df)  # <-- this steps actually executes the API call, thanks to the lazy evaluation of Spark

# COMMAND ----------

# DBTITLE 1,measuring approximate run time for the method above
end = datetime.datetime.now()
elapsed = end - start
minutes = round(elapsed.total_seconds() / 60, 3)
print("approximate run time for this method = ", minutes, " minutes")

# COMMAND ----------

# DBTITLE 1,validating the size of the output (# of groups times # of days in the year)
# getting shape for spark dataframe = rows & columns
print(("table size in rows, columns: ", grouped_spark_df.count(), len(grouped_spark_df.columns)))  # <-- should be # of groups x 365 (dates)

# COMMAND ----------

# MAGIC %md ## PySpark-only (no for-loop) approach on the larger data:

# COMMAND ----------

# DBTITLE 1,measuring start time of this method
start = datetime.datetime.now()

# COMMAND ----------

from pyspark.sql.functions import lit, col, when, sum as _sum # <-- default sum in the agg function returns an error for no good reason

# summarizing units_sold by location and date...but...
#      the output will have gaps in the dates, which we used Pandas reindex to address above
grouped_spark_df = spark_df.groupBy('location_ID', 'transaction_date').agg(_sum('units_sold').alias('units_sold'))

# to address the missing dates without Pandas...
# one option is to create an "exploded" dataframe having all locations with all (continuous, relevant) dates:
id_dates_df = grouped_spark_df.selectExpr("location_ID").distinct().selectExpr("location_ID","explode(sequence(date('2022-01-01'),date('2022-12-31'),INTERVAL 1 DAY)) as transaction_date") 

# then we do a left join of the summary units_sold onto the "exploded" dataframe having all dates...
#      with a step to replace resulting NULL units_sold with a value of zero:
grouped_spark_df = id_dates_df.alias("a").join(grouped_spark_df.alias("b"), (col("a.location_ID") == col("b.location_ID")) & (col("a.transaction_date") == col("b.transaction_date")),'left').select(col("a.location_ID"), col("a.transaction_date"), col("b.units_sold"))
# Note:  In above line, using the dataframe.alias approach to avoid this issue:  https://issues.apache.org/jira/browse/SPARK-14948
#          Tried many solutions, including renaming id columns, but using the .alias is what worked.
#          Original line of code that caused the ambiguous-join issue is shown below:
#grouped_spark_df = id_dates_df.join(grouped_spark_df, (id_dates_df.location_ID == grouped_spark_df.location_ID) & (id_dates_df.transaction_date == grouped_spark_df.transaction_date),'left').select(id_dates_df.location_ID, id_dates_df.transaction_date, grouped_spark_df.units_sold)

grouped_spark_df = grouped_spark_df.withColumn('units_sold', when(col('units_sold').isNull(), 0).otherwise(col('units_sold'))).sort('location_ID','transaction_date') 

display(grouped_spark_df)

# COMMAND ----------

# DBTITLE 1,measuring approximate run time for the method above
end = datetime.datetime.now()
elapsed = end - start
minutes = round(elapsed.total_seconds() / 60, 3)
print("approximate run time for this method = ", minutes, " minutes")

# COMMAND ----------

# DBTITLE 1,validating the size of the output (# of groups times # of days in the year)
# getting shape for spark dataframe = rows & columns
print(("table size in rows, columns: ", grouped_spark_df.count(), len(grouped_spark_df.columns)))  # <-- should be # of groups x 365 (dates)

# COMMAND ----------

# MAGIC %md ## Reviewing above run times (larger data):
# MAGIC - ### groupBy.applyinPandas = 1+ minutes
# MAGIC - ### PySpark (no Pandas) = 2+ minutes

# COMMAND ----------

# DBTITLE 1,measuring total run time of the notebook
nb_end = datetime.datetime.now()
elapsed = nb_end - nb_start
minutes = round(elapsed.total_seconds() / 60, 2)
print("total run time of notebook = ", minutes, " minutes")

# COMMAND ----------

# MAGIC %md # Conclusions:
# MAGIC > ## Beware the for-loop!
# MAGIC > ## The groupBy.applyinPandas and the PySpark-only approaches are awesome.
# MAGIC > ### The ThreadPool approach provides some reduced run time (compared to the for-loop), however...
# MAGIC >>> #### not nearly as great an improvement as seen with the more truly distributed approaches.
# MAGIC ## This notebook doesn't delve into the mechanics behind the scenes, but it all comes down to how each approach does (or does not) take advantage of Spark's super power of distributing the workload across the cluster and fully utilizing the available cores:

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=17uz9n_nkB-99iv6ANiCQtwGb9MTnwMv3' width="900" height="1300">

# COMMAND ----------

# MAGIC %md # Challenge:
# MAGIC >#### The PySpark-only approach on the larger data runs in just over 2 minutes, while...
# MAGIC >#### the groupBy.applyinPandas runs in just over 1 minute on the larger data.
# MAGIC ## Find a PySpark-only approach that runs in < 1 minute on the larger data!
# MAGIC >#### and please let me know if/when you do :-)

# COMMAND ----------

# MAGIC %md #References and additional materials:
# MAGIC - #### Sophisticated and flexible approach to generating synthetic data:  
# MAGIC >> https://databrickslabs.github.io/dbldatagen/public_docs/APIDOCS.html
# MAGIC - #### Tutorial for how to use groupBy.applyinPandas for ML modeling: 
# MAGIC >> https://github.com/marshackVB/parallel_models_blog
# MAGIC - #### More details on the concurrent.futures module and ThreadPool executor:
# MAGIC >> https://docs.python.org/3/library/concurrent.futures.html
