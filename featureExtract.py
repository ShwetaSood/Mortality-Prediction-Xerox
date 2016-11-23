

import csv
import pandas as p

colnames1 = [2,7]
colnames2 = [4,7,8,10,14,20,21]

writer = csv.writer(open('/home/iiitd/Desktop/Xerox data/featuresTrain1000.csv', 'w'))

labelData = p.read_csv('./label1Train.csv')
data1 = p.read_csv('/home/iiitd/Desktop/Xerox data/vitals1Train.csv')
data2 = p.read_csv('/home/iiitd/Desktop/Xerox data/labs1Train.csv')

uniqueId = list(set(data1.ID.tolist()))
for i in uniqueId:
	listFeatures = []
	vital_data = data1[data1['ID'] == i]
	lab_data = data2[data2['ID'] == i]

	for j in colnames1:
		dataForId = vital_data.iloc[:,j]
		if (j==2):
			s1 = round(sum(dataForId)/len(dataForId),2)
			listFeatures.append (s1)
			listFeatures.append (dataForId.iloc[-1])
		if (j== 7):
			s1 = round(sum(dataForId)/len(dataForId),2)
			listFeatures.append (s1)
	for k in colnames2:
		dataForId = lab_data.iloc[:,k]
		if(k == 4):
			d = round(sum(dataForId)/len(dataForId),2)
			listFeatures.append (d)
		if (k == 7):
			listFeatures.append (min(dataForId))
			listFeatures.append (max(dataForId))
			d = round(sum(dataForId)/len(dataForId),2)
			listFeatures.append (d)
			listFeatures.append (dataForId.iloc[-1])
		if (k == 8):
			listFeatures.append (min(dataForId))
			listFeatures.append (max(dataForId))
			d = round(sum(dataForId)/len(dataForId),2)
			listFeatures.append (d)
			listFeatures.append (dataForId.iloc[-1])
		if (k==10):
			d = round(sum(dataForId)/len(dataForId),2)
			listFeatures.append (d)
			listFeatures.append (dataForId.iloc[-1])
		if (k ==14):
			d = round(sum(dataForId)/len(dataForId),2)
			listFeatures.append (d)
		if ( k == 20):
			listFeatures.append (max(dataForId))
		if (k==21):
			d = round(sum(dataForId)/len(dataForId),2)
			listFeatures.append (d)

	label = labelData[labelData['ID'] == i].iloc[0,1]
	listFeatures.append(label)
	writer.writerow(listFeatures)
