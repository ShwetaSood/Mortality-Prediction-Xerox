

import csv
import pandas as p

#colnames1 = ['ID','V1','V2','V3','V4','V5','V6']
colnames1 = [2,7]
colnames2 = [4,7,8,10,14,20,21]


# colnames1 = range(2, 8)
# colnames2 = range(2, 22)
# colnames2 = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10','L11','L12','L13','L14','L15','L16','L17','L18','L19','L20','L21','L22','L23','L24','L25']

# reader1 = csv.reader(open('/home/iiitd/Desktop/Xerox data/label1Train.csv', 'rb'))
# reader2 = csv.reader(open('/home/iiitd/Desktop/Xerox data/vitals1Train.csv', 'rb'))
writer = csv.writer(open('/home/iiitd/Desktop/Xerox data/featuresTrain1000.csv', 'w'))


#, names=colnames1

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
