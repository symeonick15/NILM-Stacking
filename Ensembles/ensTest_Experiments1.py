from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.electric import align_two_meters
from nilmtk.disaggregate import Disaggregator
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import neural_network
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn import ensemble
from sklearn import tree
import metrics
#import metricsOnArray
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

class experimentInfo:
    def __init__(self, dsList, building, meter_key, outName, pathOrigDS, meterTH = None):
        self.dsList = dsList    # list of paths of base models output files (input for stack meta-model)
        self.building = building
        self.meter_key = meter_key
        self.outName = outName      # path for output
        self.pathOrigDS = pathOrigDS    # path of original Dataset
        self.meterTH = meterTH      # Threshold to use for current meter metrics (if None default will be used - retrieved from metadata)
        
#================= Predictions for each chunk of input file + Write them to the output (similar h5 format/structure to original dataset)
def runExperiment(experiment: experimentInfo, metricsResFileName,clearMetricsFile):
    dsPathsList_Test = experiment.dsList
    outFileName = experiment.outName
    test_building = experiment.building
    meter_key = experiment.meter_key
    pathOrigDS = experiment.pathOrigDS
    meterTH = experiment.meterTH
    print('House ',test_building)

    # Load a "complete" dataset to have the test's timerange
    test = DataSet(dsPathsList_Test[0])
    test_elec = test.buildings[test_building].elec
    testRef_meter = test_elec.submeters()[meter_key] # will be used as reference to align all meters based on this

    # Align every test meter with testRef_meter as master
    test_series_list = []
    for path in dsPathsList_Test:
        test = DataSet(path)
        test_elec = test.buildings[test_building].elec
        test_meter = test_elec.submeters()[meter_key]
        # print('Stack test: ', test_meter.get_timeframe().start.date(), " - ", test_meter.get_timeframe().end.date())
        aligned_meters = align_two_meters(testRef_meter, test_meter)
        test_series_list.append(aligned_meters)

    # Init vars for the output
    MIN_CHUNK_LENGTH = 300  # Depends on the basemodels of the ensemble
    timeframes = []
    building_path = '/building{}'.format(test_meter.building())
    mains_data_location = building_path + '/elec/meter1'
    data_is_available = False
    disag_filename = outFileName
    output_datastore = HDFDataStore(disag_filename, 'w')

    run = True
    chunkDataForOutput = None
    # -- Used to hold necessary data for saving the results using NILMTK (e.g. timeframes).
    # -- (in case where chunks have different size (not in current implementation), must use the chunk whose windowsSize is the least (to have all the data))

    while run:
        try:
            testX = []
            columnInd = 0
            # Get Next chunk of each series
            for testXGen in test_series_list:
                chunkALL = next(testXGen)
                chunk = chunkALL['slave']  # slave is the meter needed (master is only for aligning)
                chunk.fillna(0, inplace=True)
                if (columnInd == 0):
                    chunkDataForOutput = chunk  # Use 1st found chunk for it's metadata
                if (testX == []):
                    testX = np.zeros([len(chunk), len(
                        test_series_list)])  # Initialize the array that will hold all of the series as columns
                testX[:, columnInd] = chunk[:]
                columnInd += 1
            testX = scaler.transform(testX)
        except:
            run = False
            break

        if len(chunkDataForOutput) < MIN_CHUNK_LENGTH:
            continue
        # print("New sensible chunk: {}".format(len(chunk)))

        startTime = chunkDataForOutput.index[0]
        endTime = chunkDataForOutput.index[-1] # chunkDataForOutput.shape[0] - 1
        # print('Start:',startTime,'End:',endTime)
        timeframes.append(TimeFrame(startTime,endTime))  #info needed for output for use with NILMTK
        measurement = ('power','active')

        pred = clf.predict(testX)
        column = pd.Series(pred, index=chunkDataForOutput.index, name=0)
        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_power = pd.DataFrame(appliance_powers_dict)
        appliance_power[appliance_power < 0] = 0

        # Append prediction to output
        data_is_available = True
        cols = pd.MultiIndex.from_tuples([measurement])
        meter_instance = test_meter.instance()
        df = pd.DataFrame(
            appliance_power.values, index=appliance_power.index,
            columns=cols, dtype="float32")
        key = '{}/elec/meter{}'.format(building_path, meter_instance)
        output_datastore.append(key, df)

        # Append aggregate data to output
        mains_df = pd.DataFrame(chunkDataForOutput, columns=cols, dtype="float32")
        # Note (For later): not 100% right. Should be mains. But it won't be used anywhere, so it doesn't matter in this case
        output_datastore.append(key=mains_data_location, value=mains_df)

    # Save metadata to output
    if data_is_available:

        disagr = Disaggregator()
        disagr.MODEL_NAME = 'Stacked model'

        disagr._save_metadata_for_disaggregation(
            output_datastore=output_datastore,
            sample_period=sample_period,
            measurement=measurement,
            timeframes=timeframes,
            building=test_meter.building(),
            meters=[test_meter])


    #======================== Calculate Metrics =====================================
    testYDS = DataSet(pathOrigDS)
    testYDS.set_window(start=test_meter.get_timeframe().start.date(), end=test_meter.get_timeframe().end.date())
    testY_elec = testYDS.buildings[test_building].elec
    testY_meter = testY_elec.submeters()[meter_key]
    test_mains = testY_elec.mains()

    result = DataSet(disag_filename)
    res_elec = result.buildings[test_building].elec
    rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], testY_meter, meterTH, meterTH)
    relError = metrics.relative_error_total_energy(res_elec[meter_key], testY_meter)
    MAE = metrics.mean_absolute_error(res_elec[meter_key], testY_meter)
    RMSE = metrics.RMSE(res_elec[meter_key], testY_meter)
    print("============ Recall: {}".format(rpaf[0]))
    print("============ Precision: {}".format(rpaf[1]))
    print("============ Accuracy: {}".format(rpaf[2]))
    print("============ F1 Score: {}".format(rpaf[3]))
    print("============ Relative error in total energy: {}".format(relError))
    print("============ Mean absolute error(in Watts): {}".format(MAE))
    print("=== For docs: {:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}".format(rpaf[0],rpaf[1],rpaf[2],rpaf[3],relError,MAE))
    # print("============ RMSE: {}".format(RMSE))
    # print("============ TECA: {}".format(metrics.TECA([res_elec[meter_key]],[testY_meter],test_mains)))

    resDict = {'model':'TEST','building':test_building,'Appliance':meter_key, 'Appliance_Type':2,'Recall':rpaf[0],'Precision':rpaf[1],
               'Accuracy':rpaf[2],'F1':rpaf[3],'relError':relError,'MAE':MAE,'RMSE':RMSE}
    metrics.writeResultsToCSV(resDict,metricsResFileName,clearMetricsFile)

    # # trainYDS_Plot = DataSet(dsPathY) -- NILMTK PLOT FUNCTION (downsampled)--
    # testYDS.set_window(start=res_elec[meter_key].get_timeframe().start.date(), end=res_elec[meter_key].get_timeframe().end.date())
    # testY_elec = testYDS.buildings[test_building].elec
    # testY_meter = testY_elec.submeters()[meter_key]
    # res_elec[meter_key].plot()
    # testY_meter.plot()
    # plt.show()

    # # -- Normal plot --
    # chunkY = next(testY_meter.power_series(sample_period = sample_period))
    # chunk = next(res_elec[meter_key].power_series(sample_period = sample_period))
    # plt.plot(chunkY, label='ground')
    # plt.plot(chunk, label='prediction')
    # plt.legend()
    # plt.show()

def getReferenceMeterForHouse(b):
    # Load a 'complete' dataset to have as reference (for the train timerange)
    # pathsList stays the same, so no need to pass it as parameter (global) (same with meterKey)
    train = DataSet(dsPathsList[b][0])
    train_elec = train.buildings[b].elec
    train_meterRef = train_elec.submeters()[meter_key]
    return train_meterRef

def getMeterTargetGenerator(b, train_meterRef):
    trainYDS = DataSet(dsPathY)
    print('Stack train: ', train_meterRef.get_timeframe().start.date(), " - ",
          train_meterRef.get_timeframe().end.date())
    # trainYDS.set_window(start=train_meter.get_timeframe().start.date(), end=train_meter.get_timeframe().end.date())
    trainY_elec = trainYDS.buildings[b].elec
    trainY_meter = trainY_elec.submeters()[meter_key]
    # print(trainY_meter.sample_period())
    trainYGen = align_two_meters(train_meterRef, trainY_meter)
    return trainYGen

# NOTE: if not working ... return each generator separaretly
def getStackTrainGenerators(b, train_meterRef):
    trainXGen_list = []
    for path in dsPathsList[b]:
        train = DataSet(path)
        train_elec = train.buildings[b].elec
        train_meter = train_elec.submeters()[meter_key]
        # print('Stack train: ', train_meter.get_timeframe().start.date(), " - ", train_meter.get_timeframe().end.date())
        # Align the 'train_meterRef' with the X file (smaller). it's also a way to read the X meters chunk-by-chunk
        aligned_meters = align_two_meters(train_meterRef, train_meter)
        trainXGen_list.append(aligned_meters)
    return trainXGen_list

# ================================================
# =============== MAIN CODE ======================
# ================================================

#NOTE: for some reason, possibly due to NILMTK and/or generators, when I loaded all the generators needed from the beggining and added them to lists
#they would become empty (stop on first next). Because of this, I changed completely the way I handle them. Now I call every generator the moment it is
#needed. May be a little heavier due to reloading, but it works.

print("========== OPEN DATASETS ============")
dsPathY = 'ukdale.h5' # path to original Dataset (for training)
dsPathY_test = 'redd.h5' # path to original Dataset (for testing, in case it's different)
train_buildings = [1,5] #,2,3,4
sample_period = 6
meter_key = 'washing machine'
# -- Below are defined the paths for the training + testing of the meta-model (stacker) (i.e. outputs from base models)
# -- these are examples of the hierarchy used to organize the files during experiments.
# trainDSPathRoot = '/media/user/Ext hard dr/NILM/Experiments/Ukdale_1_1/' + meter_key + '/'
# testDSPathRoot = '/media/user/Ext hard dr/NILM/Experiments/Ukdale_1_'
trainDSPathRoot = '/media/user/Ext hard dr/NILM/Experiments/Ukdale_multihouse_train/' + meter_key + '/'
# testDSPathRoot = '/media/user/Ext hard dr/NILM/Experiments/Ukdale_multihouse_'
testDSPathRoot = '/media/user/Ext hard dr/NILM/Experiments/Ukdale_multihouse_REDDtest'
test_buildings = [1,2,3,4,5,6]
dsPathsList = {}  # empty dict, that will hold a list of train paths for each house
importModelName = None # filepath -- in case there is already a trained meta-model to be used (e.g. 'stackMetaModel')
experimentsList_Test = []
# --- Custom routine to easily access all the necessary datasets (works with the above hierarchy, change if files are organized differently)---
algorithmsToUse = ['DAE','GRU','RNN','SS2P','WGRU'] #'DAE','RNN','GRU','SS2P','WGRU'
for b in train_buildings:
    dsPathsList[b] = []  #init empty list for this house
    for alg in algorithmsToUse:
        dsPathsList[b].append(trainDSPathRoot + alg + '/' + 'StackTrain-h' + str(b) + '.h5')
for b in test_buildings:
    experiment = experimentInfo([], b, meter_key, 'ens-out' + str(b) + '.h5', dsPathY_test)
    for alg in algorithmsToUse:
        # experiment.dsList.append(testDSPathRoot + str(b) + '/' + meter_key + '/' + alg + '/' + 'StackTest.h5')
        experiment.dsList.append(testDSPathRoot + '/' + meter_key + '/' + alg + '/' + 'StackTest-' + str(b) + '.h5')
    experimentsList_Test.append(experiment)
#experimentsList will be a list of experimentInfo.
#Each object represents a target experiment (e.g. test for house 2) and holds the respective test datasets of this experiment, needed for the stacking metamodel.

#========================= Scale data (fit)===================================
# (Read once all the train data to fit the scaler)
# Note: there's probably no need to align meters here, because data will be read, just to have the stats for scaling calculated
scaler = StandardScaler()

for b in train_buildings:
    #Not optimal way... reloads needed meters every time, but should not be a problem
    trainX_listForScale = getStackTrainGenerators(b, getReferenceMeterForHouse(b))
    while True:
        try:
            trainX = []
            columnInd = 0
            # 1st chunk of each series + the target series
            for trainXIt in trainX_listForScale:
                chunk = next(trainXIt)
                chunk = chunk['slave']  # Only the X chunk is needed
                chunk.fillna(0, inplace=True)
                if (trainX == []):
                    trainX = np.zeros([len(chunk), len(
                        trainX_listForScale)])  # Initialize the array that will hold all of the series as columns
                trainX[:, columnInd] = chunk[:]
                columnInd += 1
        except:
            break

        scaler.partial_fit(trainX)  # fit only on training data

#========================= Train chunks ================================
if(importModelName is not None):
    clf = load(importModelName)  # load a pretrained model
else:
    # -- Some of the models used --
    # clf = linear_model.SGDRegressor()
    # clf = tree.DecisionTreeRegressor() #max_depth=5,min_samples_split=0.15, min_samples_leaf=0.09
    # clf = ensemble.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(max_depth=3), n_estimators=25, learning_rate=0.1, loss='square') #, loss='square'
    # clf = ensemble.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(), n_estimators=30, learning_rate=0.5) # min_samples_leaf=0.09
    # clf = ensemble.GradientBoostingRegressor(learning_rate=0.5, n_estimators=25) # doesnt seem to be better that AdaBooster...
    # clf = ensemble.RandomForestRegressor(n_estimators=20) # min_samples_leaf=0.1,
    clf = neural_network.MLPRegressor(100)

    # Go through chunks and feed them to the model
    trainX = []
    trainY = []
    for b in train_buildings:
        train_meterRef = getReferenceMeterForHouse(b)
        trainYGen = getMeterTargetGenerator(b, train_meterRef)
        trainXGen_list = getStackTrainGenerators(b, train_meterRef)
        run = True
        while(run):
            try:
                # Due to the way the allign_two_meters works in NILMTK, if the slave chunk is empty it will skip the whole chunk. This causes sometimes the X chunk
                # to get desync from the Y chunk. However because everything is resampled with reference to an X meter (stack train set), the X chunk will never be skipped
                # (it always exists). So to fix this problem, when len(X) != len(Y) just skip X chunks until they are equal (should work). Could have problem if:
                # produced X set filtered out some input during it's production different to other X sets OR
                # FIX: check master chunk's first timestamp (reference meter). If '=' its OK, else desynced
                trainXtmp = []
                columnInd = 0
                chunkY = next(trainYGen)
                # 1st chunk of each series + the target series
                for trainXGen in trainXGen_list:
                    chunk = next(trainXGen)
                    while(chunkY['master'].index[0] != chunk['master'].index[0]): # check if 1st timestamp is the same to sync them up (described above)
                        chunk = next(trainXGen)
                    chunk = chunk['slave']  # Only the X chunk is needed
                    chunk.fillna(0, inplace=True)
                    if (trainXtmp == []):
                        trainXtmp = np.zeros(
                            [len(chunk), len(trainXGen_list)])  # Initialize the array that will hold all of the series as columns
                    trainXtmp[:, columnInd] = chunk[:]
                    columnInd += 1
                trainXtmp = scaler.transform(trainXtmp)  # --- To SCALE the features ---
                # Also read the Y meter (target values)
                chunk = chunkY['slave']
                chunk.fillna(0, inplace=True)
                trainYtmp = np.zeros([len(chunk)])
                trainYtmp[:] = chunk[:]
                # -- Append current processed chunks to the total dataset (NOTE: train will have all data in memory, because most algorithms don't suppot partial fit) --
                if(trainX == []):
                    trainX = trainXtmp
                    trainY = trainYtmp
                else:
                    trainX = np.append(trainX, trainXtmp, axis=0)
                    trainY = np.append(trainY, trainYtmp, axis=0)
            except:
                run = False
                break

            print("Train chunk size: {}".format(len(chunk)))
            startTime = chunk.index[0]
            endTime = chunk.index[-1]
            print('Start:', startTime, 'End:', endTime)

    # -- After all data were read, train the model --
    clf.fit(trainX, trainY)

    dump(clf, 'stackMetaModel')

# === Do the tests (predictions) ====
metricsFileName = 'testResults.csv'
clearMetricsFile = True #Set to true if want to clear metrics file, before starting the experiments (else the results will be appended)
for experiment in experimentsList_Test:
    runExperiment(experiment, metricsFileName, clearMetricsFile)
    clearMetricsFile = False

# === Overall metrics (Generalization, Four appliance type, etc.) ====
# print('GoUH: {}'.format(metrics.GeneralizationOverUnseenHouses_fromCSV(metricsFileName,'F1')))