from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import pyspark.sql.functions as F
import pyspark.sql.types as T
import statistics
import datetime
import json
import numpy as np
from pyspark.sql.functions import concat, lit

def main(sc, spark):

  dfPlaces = spark.read.csv('/data/share/bdm/core-places-nyc.csv', header=True, escape='"')
  dfPattern = spark.read.csv('/data/share/bdm/weekly-patterns-nyc-2019-2020/*', header=True, escape='"')

  CAT_CODES = ['452210', '452311', '445120', '722410', '722511', '722513', '446110', '446191', '311811', '722515', '445210', '445220', '445230', '445291', '445292', '445299', '445110']
  CAT_GROUP = [0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 8]
  CAT_GROUP = dict(zip(CAT_CODES, CAT_GROUP))

  dfD = dfPlaces \
      .filter(F.col('naics_code').isin(*CAT_GROUP)) \
      .select('placekey', 'naics_code')
  udfToGroup = F.udf(lambda x: CAT_GROUP[x], T.IntegerType())
  dfE = dfD.withColumn('group', udfToGroup('naics_code'))
  dfF = dfE.drop('naics_code').cache()

  dfgroup = dfF.groupBy("group").count().take(9)
  groupCount = {}
  for i in dfgroup:
    groupCount[i[0]] = i[1]

  def expandVisits(date_range_start, visits_by_day):
      start = datetime.date(*map(int, date_range_start[:10].split('-')))
      visits = json.loads(visits_by_day)
      finalList = []
      for d, cnt in enumerate(visits):
        date = start + datetime.timedelta(days=d)
        finalList.append((date.year, date.strftime("%m-%d"), cnt))
      return finalList

  visitType = T.StructType([T.StructField('year', T.IntegerType()),
                            T.StructField('date', T.StringType()),
                            T.StructField('visits', T.IntegerType())])

  udfExpand = F.udf(expandVisits, T.ArrayType(visitType))

  dfH = dfPattern.join(dfF, 'placekey') \
      .withColumn('expanded', F.explode(udfExpand('date_range_start', 'visits_by_day'))) \
      .select('group', 'expanded.*')

  def computeStats(group, visits):
      totalCount = groupCount[group]
      visits += [0]*(totalCount-len(visits))
      median = int(statistics.median(visits)+0.5)
      stdev = int(statistics.stdev(visits)+0.5)
      low = max(0, median-stdev)
      high = median + stdev
      return {'median': median, 'low': low, 'high': high}

  statsType = T.StructType([T.StructField('median', T.IntegerType()),
                            T.StructField('low', T.IntegerType()),
                            T.StructField('high', T.IntegerType())])

  udfComputeStats = F.udf(computeStats, statsType)

  dfI = dfH.groupBy('group', 'year', 'date') \
      .agg(F.collect_list('visits').alias('visits')) \
      .withColumn('stats', udfComputeStats('group', 'visits'))

  dfJ = dfI \
    .select('group', 'year', 'date', 'stats.*', lit('2020-').alias('prefix'))
  dfJ = dfJ.select('group', 'year', concat(dfJ.prefix, dfJ.date).alias('date'), 'median', 'low', 'high')

  dfJ = dfJ.where('year >= 2019') \
    .sort('group', 'year', 'date') \
    .cache

  CAT_LABEL = ['big_box_grocers', 'convenience_stores', 'drinking_places', 'full_service_restaurants', 'limited_service_restaurants', 'pharmacies_and_drug_stores', 'snack_and_retail_bakeries', 'specialty_food_stores', 'supermarkets_except_convenience_stores']

  OUTPUT_PREFIX = '/content'
  toFileName = lambda x:'_'.join((''.join(map(lambda c: c if c.isalnum() else ' ', x.lower()))).split())
  for i,filename in enumerate(map(toFileName, CAT_LABEL)):
      dfJ.filter(dfJ.group == i) \
      .drop('group') \
      .coalesce(1) \
      .write.csv(f'{OUTPUT_PREFIX}/{filename}',
                 mode='overwrite', header=True)

if __name__=='__main__':
    sc = SparkContext()
    spark = SparkSession(sc)
    main(sc, spark)
