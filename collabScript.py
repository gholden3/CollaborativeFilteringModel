from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import csv
import pickle

sc = SparkContext(appName="CollabFilter")

#hash the user IDS in AskForRecsFor file to make them ints
try:
    longToShortUsers = pickle.load( open("ShortenUserIDDict.p", "rb" ) ) 
    numShorts = len(longToShortUsers)
except (OSError, IOError) as e: 
    longToShortUsers = dict()
    numShorts = 0
fp = open("AskForRecsForLong.csv","r")
fpout = open("AskForRecsForShort.csv","wb")
csv_f = csv.reader(fp)
for row in csv_f:
    longUserID = row[0]
    numRecs = row[1]
    if longUserID in longToShortUsers:
        shortUserID = longToShortUsers[longUserID]
    else:
        shortUserID = str(numShorts+1)
        numShorts += 1
        longToShortUsers[str(longUserID)] = shortUserID           
    outStr = str(shortUserID) + "," + str(numRecs) + "\n"
    fpout.write(outStr)
pickle.dump( longToShortUsers, open( "ShortenUserIDDict.p", "wb" ))
fp.close()
fpout.close()

#hash locations in visits file to make them ints
try:
    longToShortLocations = pickle.load( open("ShortenLocations.p", "rb" ) )
    numLocations = len(longToShortLocations)
except (OSError, IOError) as e:
    longToShortLocations = dict()
    numLocations = 0
longToShortUsers = pickle.load( open("ShortenUserIDDict.p", "rb" ) )
fp = open("RealVisitsData.csv","r")
fpout = open("RealVisitsDataShort.csv","wb")
csv_f = csv.reader(fp)
for row in csv_f:
    longUserID = row[0]
    longLocation = row[1]
    numVisits = row[2]
    shortUserID = longToShortUsers[longUserID]
    if longLocation in longToShortLocations:
        shortLocation = longToShortLocations[longLocation]
    else:
        shortLocation = str(numLocations +1)
        numLocations += 1
        longToShortLocations[str(longLocation)] = shortLocation
    outStr = str(shortUserID) + "," + str(shortLocation) + "," + numVisits + "\n"
    fpout.write(outStr)
pickle.dump( longToShortLocations, open( "shortenLocations.p", "wb" ))
fp.close()
fpout.close()


# Load and parse the data
data = sc.textFile("file:///home/hadoop/RealVisitsDataShort.csv")
ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.trainImplicit(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

#Save and load model
#commented out the save for now because the model already exists on hdfs
#uncomment this when you are ready to train a new model!
#model.save(sc, "target/tmp/myCollaborativeFilter")
sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
#parse the AskForRecsFor.csv file
f = open('AskForRecsForShort.csv')
fp = open("reccomendFile2.txt","w")
csv_f = csv.reader(f)
#next(csv_f, None) 
for row in csv_f:
   a = row[0]
   b = row[1]
   recommendation = model.recommendProducts(int(a),int(b))
   fp.write(str(recommendation))
fp.close()
