# -*- coding: utf-8 -*-
"""
    Calculator
    ~~~~~~~~~~~~~~

    A simple Calculator made by Flask and jQuery.

    :copyright: (c) 2015 by Grey li.
    :license: MIT, see LICENSE for more details.
"""
import re
from flask import Flask, jsonify, render_template, request
app = Flask(__name__)


import findspark
findspark.init()
from pyspark.context import SparkContext
from pyspark.sql import Row
from pyspark.sql.session import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
sc = SparkContext('local')
spark = SparkSession(sc)
import pandas as pd


#model training part:
lines = spark.read.text("/Users/junyan/Downloads/properties_processed.csv").rdd
parts = lines.map(lambda row: row.value.split(","))
#priceRDD = parts.map(lambda p:Row(aa = float(p[0]),parceled = float(p[1]), airconditioningtypeid = float(p[2]),architecturalstyletypeid = float(p[3]),basementsqft = float(p[4]), bathroomcnt = float(p[5]),bedroomcnt = float(p[6]),buildingqualitytypeid = float(p[7]),calculatedbathnbr = float(p[8]),decktypeid = float(p[9]),calculatedfinishedsquarefeet = float(p[10]),finishedsquarefeet15 = float(p[11]),fips = float(p[12]),fireplacecnt = float(p[13]),fullbathcnt = float(p[14]),garagecarcnt = float(p[15]),garagetotalsqft = float(p[16]),hashottuborspa = float(p[17]),heatingorsystemtypeid = float(p[18]),latitude = float(p[19]),longitude = float(p[20]),lotsizesquarefeet = float(p[21]),poolcnt = float(p[22]),poolsizesum = float(p[23]),propertylandusetypeid = float(p[24]),rawcensustractandblock = float(p[25]),regionidcounty = float(p[26]),regionidzip = float(p[27]),threequarterbathnbr = float(p[28]),unitcnt = float(p[29]),yardbuildingsqft17 = float(p[30]),yardbuildingsqft26 = float(p[31]),yearbuilt = float(p[32]),numberofstories = float(p[33]),fireplaceflag = float(p[34]),assessmentyear = float(p[35]),taxamount = float(p[36]),taxdelinquencyflag = float(p[37]),zvalue = float(p[38])))

priceRDD = parts.map(lambda p:Row(airconditioningtypeid = float(p[2]),bathroomcnt = float(p[5]),bedroomcnt = float(p[6]),buildingqualitytypeid = float(p[7]),calculatedbathnbr = float(p[8]),decktypeid = float(p[9]),calculatedfinishedsquarefeet = float(p[10]),finishedsquarefeet15 = float(p[11]),fireplacecnt = float(p[13]),garagecarcnt = float(p[15]),garagetotalsqft = float(p[16]),heatingorsystemtypeid = float(p[18]),latitude = float(p[19]),longitude = float(p[20]),poolcnt = float(p[22]),poolsizesum = float(p[23]),propertylandusetypeid = float(p[24]),regionidcounty = float(p[26]),threequarterbathnbr = float(p[28]),unitcnt = float(p[29]),yearbuilt = float(p[32]),taxamount = float(p[36]),zvalue = float(p[38])))



price_f = priceRDD.map(lambda p:Row(label=(float(p[22])),features = Vectors.dense(p[0:22])))
price = price_f.toDF()

(train, test) = price.randomSplit([0.7,0.3])
rf = RandomForestRegressor(maxDepth=15, maxBins=26, minInstancesPerNode=1, minInfoGain=0.0,numTrees=20)

global model

model = rf.fit(train)



############################################



@app.route('/')
def index():
    return render_template('r.html')

@app.route('/rate', methods=['POST'])
def rate():
  
  global model
  #context=dict()
  data = ['a']*38
  for i in range(1,11):
      data[i-1] = request.form['feature%d'%i]

  if request.form['other'] != '':
      temp = request.form['other'].split(',')
      for j in range(len(temp)):
          data[10+j] = temp[j]
  
  
  properties = data
  properties[10] = data[0]
  properties[6] = data[1]
  properties[5] = data[2]
  properties[22] = data[3]
  properties[32] = data[4]
  properties[29] = data[5]
  properties[16] = data[6]
  properties[18] = data[7]
  properties[24] = data[8]
  properties[19] = data[9]
  properties[0] = data[10]
  properties[1] = data[11]
  properties[2] = data[12]
  properties[3] = data[13]
  properties[4] = data[14]
  properties[7] = data[15]
  properties[8] = data[16]
  properties[9] = data[17]
  properties[11] = data[18]
  properties[12] = data[19]
  properties[13] = data[20]
  properties[14] = data[21]
  properties[15] = data[22]
  properties[17] = data[23]
  properties[20] = data[24]
  properties[21] = data[25]
  properties[23] = data[26]
  properties[25] = data[27]
  properties[26] = data[28]
  properties[27] = data[29]
  properties[28] = data[30]
  properties[30] = data[31]
  properties[31] = data[32]  
     
  properties_default = [2,11000000,0,0,0,2,3,6,1,0,3000,3000,6037,0,1,1,400,-1,0,34000000,-120000000,8000,0,0,47,60400000,3101,96400,0,2,0,0,1950,1,-1,2015,5000,-1]
  
  
  
  for i in range(38):
      if properties[i]=='a' or properties[i]=='':
          properties[i] = properties_default[i]
    
  for i in range(len(properties)):
      properties[i] = float(properties[i])
      
  properties.append(float(10000))
  
 
  properties2 = []
  properties2.append(properties)
    
  test = pd.DataFrame(data=properties2)
  test.to_csv('/Users/junyan/Downloads/test1.csv',header=False, index = False)
    
  lines = spark.read.text("/Users/junyan/Downloads/test1.csv").rdd
    
  parts = lines.map(lambda row: row.value.split(","))
    
  testRDD = parts.map(lambda p:Row(airconditioningtypeid = float(p[2]),bathroomcnt = float(p[5]),bedroomcnt = float(p[6]),buildingqualitytypeid = float(p[7]),calculatedbathnbr = float(p[8]),decktypeid = float(p[9]),calculatedfinishedsquarefeet = float(p[10]),finishedsquarefeet15 = float(p[11]),fireplacecnt = float(p[13]),garagecarcnt = float(p[15]),garagetotalsqft = float(p[16]),heatingorsystemtypeid = float(p[18]),latitude = float(p[19]),longitude = float(p[20]),poolcnt = float(p[22]),poolsizesum = float(p[23]),propertylandusetypeid = float(p[24]),regionidcounty = float(p[26]),threequarterbathnbr = float(p[28]),unitcnt = float(p[29]),yearbuilt = float(p[32]),taxamount = float(p[36]),zvalue = float(p[38])))
    
  test_f = testRDD.map(lambda p:Row(label=(float(p[22])),features = Vectors.dense(p[0:22])))
  test = test_f.toDF()
  
  predictions = model.transform(test)
  predictions = predictions.rdd
  #rcmd = properties
  result = predictions.first()
  rcmd = result['prediction']
  # return render_template("start.html")
  return render_template('rate.html',data=rcmd)


if __name__ == '__main__':
    app.run()
