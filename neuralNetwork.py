#!/usr/bin/python
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter


ds = ClassificationDataSet(79, 1, nb_classes=2)
tf = open('./finalFeaturestraining.csv','rb')
for line in tf.readlines():
    data = [float(x) for x in line.strip().split(',') if x != '']
    indata =  tuple(data[:79])
    # print len(indata)
    outdata = tuple(data[79:80])
    # print len(outdata)
    ds.addSample(indata,outdata)
tstdata, trndata = ds.splitWithProportion( 0.0 )

n = buildNetwork(trndata.indim,30,30,trndata.outdim,recurrent=True)
t = BackpropTrainer(n, dataset=trndata, learningrate=0.001,momentum=0.3,verbose=True)
t.trainEpochs( 100 )
trnresult = percentError( t.testOnClassData(), trndata['class'] )
# tstresult = percentError( t.testOnClassData( dataset=tstdata ), tstdata['class'] )

print "epoch: %4d" % t.totalepochs, \
          "  train error: %5.2f%%" % trnresult

NetworkWriter.writeToFile(n, './validationData/NeuraLNetworkModelv2.xml')


