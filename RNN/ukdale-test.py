from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from rnndisaggregator import RNNDisaggregator
import metrics

print("========== OPEN DATASETS ============")
meterList = []
mainsList = []
test = DataSet('ukdale.h5')
# test = DataSet('redd.h5')
# test.set_window(start='2016-04-01',end='2016-05-01')
test_building_list = [2,3,4,5] #[2,5]
sample_period = 6
meter_key = 'kettle'

file = open('baseTrainSetsInfo_' + meter_key, 'r')
for line in file:
    toks = line.split(',')
    train = DataSet(toks[0])
    print(toks[2],'-',toks[3])
    train.set_window(start=toks[2], end=toks[3])
    train_elec = train.buildings[int(toks[1])].elec
    meterList.append(train_elec.submeters()[meter_key])
    mainsList.append(train_elec.mains())

disaggregator = RNNDisaggregator()

start = time.time()
print("========== TRAIN ============")
epochs = 0
epochsPerCheckpoint = 5
totalCheckpoints = 1 #3
# disaggregator.import_model("RNN-MultiHouse-{}-{}epochs.h5".format(meter_key,
#                                                                       epochs))
for i in range(totalCheckpoints):
    print("CHECKPOINT {}".format(epochs))
    disaggregator.train_across_buildings(mainsList, meterList, epochs=epochsPerCheckpoint, sample_period=sample_period, batch_size=256)
    epochs += epochsPerCheckpoint
    disaggregator.export_model("RNN-MultiHouse-{}-{}epochs.h5".format(meter_key,
                                                                      epochs))
end = time.time()
print("Train =", end-start, "seconds.")

file = open('stackTrainSetsInfo_' + meter_key, 'r')
for line in file:
    toks = line.split(',')
    StackTrain = DataSet(toks[0])
    print(toks[2],'-',toks[3])
    StackTrain.set_window(start=toks[2], end=toks[3])
    test_elec = StackTrain.buildings[int(toks[1])].elec
    test_mains = test_elec.mains()

    print("========== DISAGGREGATE (stackTrain)============")
    disag_filename = "StackTrain-h" + toks[1] + ".h5"
    output = HDFDataStore(disag_filename, 'w')
    disaggregator.disaggregate(test_mains, output, test_elec[meter_key], sample_period=sample_period)
    output.close()

for i in test_building_list:
    test_elec = test.buildings[i].elec
    test_mains = test_elec.mains()

    print("========== DISAGGREGATE ============")
    disag_filename = "StackTest-" + str(i) + ".h5"
    output = HDFDataStore(disag_filename, 'w')
    disaggregator.disaggregate(test_mains, output, test_elec[meter_key], sample_period=sample_period)
    output.close()

    print("========== RESULTS ============")
    result = DataSet(disag_filename)
    res_elec = result.buildings[i].elec
    rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_elec[meter_key])
    print("============ Recall: {}".format(rpaf[0]))
    print("============ Precision: {}".format(rpaf[1]))
    print("============ Accuracy: {}".format(rpaf[2]))
    print("============ F1 Score: {}".format(rpaf[3]))

    print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec[meter_key], test_elec[meter_key])))
    print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec[meter_key], test_elec[meter_key])))