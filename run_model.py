
import sys
from numpy import arange,array,ones#,random,linalg
from pylab import plot,show
from scipy import stats
import csv
import pandas as p
import math
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkreader import NetworkReader

#colnames1 = ['ID','V1','V2','V3','V4','V5','V6']
colnames1 = range(2, 8)
colnames2 = range(2, 27)
# colnames2 = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10','L11','L12','L13','L14','L15','L16','L17','L18','L19','L20','L21','L22','L23','L24','L25']

vitalsFile=sys.argv[1]
labsFile=sys.argv[2]
ageFile=sys.argv[3]


# reader1 = csv.reader(open('/home/iiitd/Desktop/Xerox data/label1Train.csv', 'rb'))
# readLabels = p.read_csv('./id_label_val.csv')
writerOutput = csv.writer(open('./output.csv', 'w'))
readerlab = csv.reader(open(labsFile, 'r'))
readervital = csv.reader(open(vitalsFile, 'r'))


listIgnoredColumns =[21, 27, 31, 33, 34, 38, 43, 53, 58, 63, 73, 78, 83, 90, 91, 93, 94, 99, 101, 103, 105, 110, 111, 113, 114, 118, 121, 122, 123, 131, 138, 143, 145, 146, 147, 149, 151, 153, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152]

labs=[7.4,40,87.5,140,4,25,4.8,0.9,7.25,40,275,1.1,1400,190,12.15,0.01,0.01,85,85,4.4,9.55,95.5,30,190,2.95]
vitals=[120,80,72,45,97,98.6]

rowL= next(readerlab)
rowV= next(readervital)

fileLab = open('./labsFeatures.csv', 'a')
fileVitals = open('./vitalsFeatureas.csv', 'a')

writerlab = csv.writer(fileLab)
writervital = csv.writer(fileVitals)

writervital.writerow(rowV)
writerlab.writerow(rowL)
fileVitals.close()
fileLab.close()

for rowVital in readervital:
	rowLab = next(readerlab)


readerlab = csv.reader(open(labsFile, 'r'))
readervital = csv.reader(open(vitalsFile, 'r'))

rowL= next(readerlab)
rowV= next(readervital)

for rowVital in readervital:
	rowLab = next(readerlab)
	for a1 in colnames2:
		if (rowLab[a1] == 'NA'):
			rowLab[a1]= str(labs[a1-2])
	for a2 in colnames1:
		if (rowVital[a2] == 'NA'):
			rowVital[a2]= str(vitals[a2-2])


	fileLab = open('./labsFeatures.csv', 'a')
	fileVitals = open('./vitalsFeatureas.csv', 'a')

	writerlab = csv.writer(fileLab)
	writervital = csv.writer(fileVitals)

	writervital.writerow(rowVital)
	writerlab.writerow(rowLab)
	fileVitals.close()
	fileLab.close()

	#now reading from the new features extracted file
	fileLab2 = open('./labsFeatures.csv')
	fileVitals2 = open('./vitalsFeatureas.csv')


	data2 = p.read_csv(fileLab2)
	data1 = p.read_csv(fileVitals2)

	fileFeatures = open('./featuresValidation.csv', 'w')
	writer = csv.writer(fileFeatures)


	uniqueId = list(set(data1.ID.tolist()))
	for i in uniqueId:
		listFeatures = []
		vital_data = data1[data1['ID'] == i]
		lab_data = data2[data2['ID'] == i]
		time=vital_data.iloc[:,1]
		a= time/3600
		for j in colnames1:
			dataForId = vital_data.iloc[:,j]
			slope, intercept, r_value, p_value, std_err = stats.linregress(a,dataForId)
			if(not(math.isnan(slope))):
				deg_slope=math.atan(slope)
			else:
				#print "Am nan"
				deg_slope=math.pi/2
			listFeatures.append (str(round(min(dataForId),2)))
			listFeatures.append (str(round(max(dataForId),2)))
			listFeatures.append (str(round(sum(dataForId)/len(dataForId),2)))
		 	listFeatures.append (str(round(dataForId.iloc[-1],2)))
		 	listFeatures.append(str(round(deg_slope,4)))	
		for k in colnames2:
			dataForId = lab_data.iloc[:,k]
			slope, intercept, r_value, p_value, std_err = stats.linregress(a,dataForId)
			if(not(math.isnan(slope))):
				deg_slope=math.atan(slope)
			else:
				deg_slope=math.pi/2
			listFeatures.append (str(round(min(dataForId),2)))
			listFeatures.append (str(round(max(dataForId),2)))
			listFeatures.append (str(round(sum(dataForId)/len(dataForId),2)))
		 	listFeatures.append (str(round(dataForId.iloc[-1],2)))
		 	listFeatures.append(str(round(deg_slope,4)))

		finalFeaturesTesting=[]
		for column in range(0,155):
			if column not in listIgnoredColumns:
				finalFeaturesTesting.append(listFeatures[column])
 	

		writer.writerow(finalFeaturesTesting)
	
	fileFeatures.close()
	fnn = NetworkReader.readFrom('./NeuraLNetworkModelv2.xml')
	lines = open("./featuresValidation.csv").readlines()[-1:]
	for e in lines:
		data = [float(x) for x in e.strip().split(',') if x != '']
	if rowVital[-1] == '1':
		predictedResult = fnn.activate(data)
		outputValueTrain = 0
		if(predictedResult[0] > 0.0389547):
			outputValueTrain= 1

		listOutput=[]
		listOutput.append(str(rowVital[0])) #ID
		listOutput.append(str(rowVital[1])) #timestamp
		listOutput.append(str(outputValueTrain)) #predicted
		writerOutput.writerow(listOutput)

	fileLab2.close()	
	fileVitals2.close()