from __future__ import print_function, division
from warnings import warn, filterwarnings

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py
import random
import sys

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout
from keras.utils import plot_model

from nilmtk.timeframe import merge_timeframes, TimeFrame
from datetime import datetime

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class DAEDisaggregator(Disaggregator):
    '''Denoising Autoencoder disaggregator from Neural NILM
    https://arxiv.org/pdf/1507.06594.pdf

    Attributes
    ----------
    model : keras Sequential model
    sequence_length : the size of window to use on the aggregate data
    mmax : the maximum value of the aggregate data

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

    def __init__(self, sequence_length):
        '''Initialize disaggregator

        Parameters
        ----------
        sequence_length : the size of window to use on the aggregate data
        meter : a nilmtk.ElecMeter meter of the appliance to be disaggregated
        '''
        self.MODEL_NAME = "AUTOENCODER"
        self.mmax = None
        self.sequence_length = sequence_length
        self.MIN_CHUNK_LENGTH = sequence_length
        self.model = self._create_model(self.sequence_length)

    def train(self, mains, meter, epochs=1, batch_size=16, **load_kwargs):
        '''Train

        Parameters
        ----------
        mains : a nilmtk.ElecMeter object for the aggregate data
        meter : a nilmtk.ElecMeter object for the meter data
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = meter.power_series(**load_kwargs)

        # Train chunks
        run = True
        mainchunk = next(main_power_series)
        meterchunk = next(meter_power_series)
        if self.mmax == None:
            self.mmax = mainchunk.max()

        import matplotlib.pyplot as plt
        while(run):
            # plt.plot(meterchunk)

            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunk = self._normalize(meterchunk, self.mmax)

            self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
            try:
                mainchunk = next(main_power_series)
                meterchunk = next(meter_power_series)
            except:
                run = False
        # plt.show()

    def trainOnActivations(self, mains, meter, minOff, minOn, epochs=1, batch_size=16, **load_kwargs):
        # ------------
        # --- experimental - not used -----
        # -------------
        minBorder = 500

        main_power_series = mains.power_series(**load_kwargs)
        mains_all = next(main_power_series)
        self.mmax = mains_all.max()

        activs = meter.activation_series(minOff, minOn, 0, meter.on_power_threshold())

        totalActivs = len(activs)
        print('Total activations', totalActivs)
        sizeSum = 0
        # find avg timeframe size of activations
        for act in activs:
            # print(act.shape[0])
            sizeSum += act.shape[0]

        border =  int((sizeSum / totalActivs) / 2)
        print(border)
        if(border < minBorder): border = minBorder
        # border = minBorder

        activs = meter.activation_series(minOff, minOn, border, meter.on_power_threshold())

        n = min(len(activs), 500)

        # firstChunk = True
        for i in range(len(activs)):
            # print(activs[i].index[0].strftime('%Y-%m-%d %H:%M:%S')," - ",activs[i].index[-1].strftime('%Y-%m-%d %H:%M:%S'))
            # -- First implementation: On disk ... ---
            # mainsDS.set_window(start=activs[i].index[0].strftime('%Y-%m-%d %H:%M:%S'), end=activs[i].index[-1].strftime('%Y-%m-%d %H:%M:%S'))
            # train_elec = mainsDS.buildings[1].elec
            # train_mains = train_elec.mains()
            # main_power_series = train_mains.power_series(**load_kwargs)
            # mains_chunk = next(main_power_series)
            # if firstChunk:
            #     self.mmax = mains_chunk.max()
            #     firstChunk = False
            mains_chunk = mains_all[activs[i].index[0]: activs[i].index[-1]]

            mains_chunk = self._normalize(mains_chunk, self.mmax)
            meter_chunk = self._normalize(activs[i], self.mmax)
            self.train_on_chunk(mains_chunk, meter_chunk, epochs, batch_size)


    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):
        '''Train using only one chunk

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        '''

        s = self.sequence_length
        #up_limit =  min(len(mainchunk), len(meterchunk))
        #down_limit =  max(len(mainchunk), len(meterchunk))

        # Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]

        # Create array of batches
        #additional = s - ((up_limit-down_limit) % s)
        additional = s - (len(ix) % s)
        X_batch = np.append(mainchunk, np.zeros(additional))
        Y_batch = np.append(meterchunk, np.zeros(additional))

        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s, 1))
        Y_batch = np.reshape(Y_batch, (int(len(Y_batch) / s), s, 1))

        self.model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=epochs, shuffle=True)

    def train_across_buildings(self, mainlist, meterlist, epochs=1, batch_size=128, **load_kwargs):
        assert(len(mainlist) == len(meterlist), "Number of main and meter channels should be equal")
        num_meters = len(mainlist)

        mainps = [None] * num_meters
        meterps = [None] * num_meters
        mainchunks = [None] * num_meters
        meterchunks = [None] * num_meters

        for i,m in enumerate(mainlist):
            mainps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(meterlist):
            meterps[i] = m.power_series(**load_kwargs)

        for i in range(num_meters):
            mainchunks[i] = next(mainps[i])
            meterchunks[i] = next(meterps[i])
        if self.mmax == None:
            self.mmax = max([m.max() for m in mainchunks])


        run = True
        while(run):
            mainchunks = [self._normalize(m, self.mmax) for m in mainchunks]
            meterchunks = [self._normalize(m, self.mmax) for m in meterchunks]

            self.train_across_buildings_chunk_CONCAT_method(mainchunks, meterchunks, epochs, batch_size)
            try:
                for i in range(num_meters):
                    mainchunks[i] = next(mainps[i])
                    meterchunks[i] = next(meterps[i])
            except:
                run = False

    # Below are different implementations of a function with the same purpose.
    # 'train_across_buildings_chunk_CONCAT_method' was actually used for the final experiments

    def train_across_buildings_chunk(self, mainchunks, meterchunks, epochs, batch_size):
        num_meters = len(mainchunks)
        batch_size = int(batch_size/num_meters)
        num_of_batches = [None] * num_meters
        s = self.sequence_length
        for i in range(num_meters):
            mainchunks[i].fillna(0, inplace=True)
            meterchunks[i].fillna(0, inplace=True)
            ix = mainchunks[i].index.intersection(meterchunks[i].index)
            m1 = mainchunks[i]
            m2 = meterchunks[i]
            mainchunks[i] = m1[ix]
            meterchunks[i] = m2[ix]

            num_of_batches[i] = int(len(ix)/(s*batch_size)) - 1

        for e in range(epochs):
            print(e)
            batch_indexes = list(range(min(num_of_batches))) #"Wrap" with list to support python 3.x
            random.shuffle(batch_indexes)

            for bi, b in enumerate(batch_indexes):

                print("Batch {} of {}".format(bi,num_of_batches), end="\r")
                sys.stdout.flush()
                X_batch = np.empty((batch_size*num_meters, s, 1))
                Y_batch = np.empty((batch_size*num_meters, s, 1))

                for i in range(num_meters):
                    mainpart = mainchunks[i]
                    meterpart = meterchunks[i]
                    mainpart = np.array(mainpart[b*batch_size*s:(b+1)*batch_size*s])
                    meterpart = np.array(meterpart[b*batch_size*s:(b+1)*batch_size*s])
                    X = np.reshape(mainpart, (batch_size, s, 1))
                    Y = np.reshape(meterpart, (batch_size, s, 1))

                    X_batch[i*batch_size:(i+1)*batch_size] = np.array(X)
                    Y_batch[i*batch_size:(i+1)*batch_size] = np.array(Y)

                p = np.random.permutation(len(X_batch))
                X_batch, Y_batch = X_batch[p], Y_batch[p]

                self.model.train_on_batch(X_batch, Y_batch)
            print("\n")

    def train_across_buildings_chunk_2(self, mainchunks, meterchunks, epochs, batch_size):
        num_meters = len(mainchunks)
        batch_size = int(batch_size/num_meters)
        num_of_batches = [None] * num_meters
        s = self.sequence_length
        for i in range(num_meters):
            mainchunks[i].fillna(0, inplace=True)
            meterchunks[i].fillna(0, inplace=True)
            ix = mainchunks[i].index.intersection(meterchunks[i].index)
            m1 = mainchunks[i]
            m2 = meterchunks[i]
            mainchunks[i] = m1[ix]
            meterchunks[i] = m2[ix]

            num_of_batches[i] = int(len(ix)/(s*batch_size)) - 1

        batch_indexes = list(range(min(num_of_batches))) #"Wrap" with list to support python 3.x
        random.shuffle(batch_indexes)

        X_batch_total = None
        Y_batch_total = None

        for bi, b in enumerate(batch_indexes):

            print("Batch {} of {}".format(bi,num_of_batches), end="\r")
            sys.stdout.flush()
            X_batch = np.empty((batch_size*num_meters, s, 1))
            Y_batch = np.empty((batch_size*num_meters, s, 1))

            for i in range(num_meters):
                mainpart = mainchunks[i]
                meterpart = meterchunks[i]
                mainpart = np.array(mainpart[b*batch_size*s:(b+1)*batch_size*s])
                meterpart = np.array(meterpart[b*batch_size*s:(b+1)*batch_size*s])
                X = np.reshape(mainpart, (batch_size, s, 1))
                Y = np.reshape(meterpart, (batch_size, s, 1))

                X_batch[i*batch_size:(i+1)*batch_size] = np.array(X)
                Y_batch[i*batch_size:(i+1)*batch_size] = np.array(Y)

            p = np.random.permutation(len(X_batch))
            X_batch, Y_batch = X_batch[p], Y_batch[p]

            if X_batch_total is None:
                X_batch_total = X_batch
                Y_batch_total = Y_batch
            else:
                X_batch_total = np.append(X_batch_total, X_batch, 0)
                Y_batch_total = np.append(Y_batch_total, Y_batch, 0)

        self.model.fit(X_batch_total, Y_batch_total,batch_size=batch_size, epochs=epochs, shuffle=False)

    def train_across_buildings_chunk_CONCAT_method(self, mainchunks, meterchunks, epochs, batch_size):
        num_meters = len(mainchunks)
        s = self.sequence_length
        X_batch= None
        Y_batch = None
        for i in range(num_meters):
            mainchunks[i].fillna(0, inplace=True)
            meterchunks[i].fillna(0, inplace=True)
            ix = mainchunks[i].index.intersection(meterchunks[i].index)
            m1 = mainchunks[i]
            m2 = meterchunks[i]
            mainchunks[i] = np.array(m1[ix])
            meterchunks[i] = np.array(m2[ix])

            if X_batch is None:
                X_batch = mainchunks[i]
                Y_batch = meterchunks[i]
            else:
                X_batch = np.append(X_batch, mainchunks[i], 0)
                Y_batch = np.append(Y_batch, meterchunks[i], 0)
            # Clear up memory ()
            mainchunks[i] = None
            meterchunks[i] = None

        additional = s - (len(X_batch) % s)
        X_batch = np.append(X_batch, np.zeros(additional))
        Y_batch = np.append(Y_batch, np.zeros(additional))

        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s, 1))
        Y_batch = np.reshape(Y_batch, (int(len(Y_batch) / s), s, 1))

        self.model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=epochs, shuffle=True)

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt.

        Parameters
        ----------
        mains : nilmtk.ElecMeter
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        meter_metadata : metadata for the produced output
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.mmax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )
            # timezone = 'Europe/London'


    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series to disaggregate
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        s = self.sequence_length
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        additional = s - (up_limit % s)
        X_batch = np.append(mains, np.zeros(additional))
        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s ,1))

        pred = self.model.predict(X_batch)
        pred = np.reshape(pred, (up_limit + additional))[:up_limit]
        column = pd.Series(pred, index=mains.index, name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers


    def import_model(self, filename):
        '''Loads keras model from h5

        Parameters
        ----------
        filename : filename for .h5 file

        Returns: Keras model
        '''
        self.model = load_model(filename)
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('mmax')
            self.mmax = np.array(ds)[0]

    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data = [self.mmax])

    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * mmax
        return tchunk

    def _create_model(self, sequence_len):
        '''Creates the Auto encoder module described in the paper
        '''
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_len, 1), padding="same", strides=1))
        model.add(Flatten())

        # Fully Connected Layers
        model.add(Dropout(0.2))
        model.add(Dense((sequence_len-0)*8, activation='relu'))

        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))

        model.add(Dropout(0.2))
        model.add(Dense((sequence_len-0)*8, activation='relu'))

        model.add(Dropout(0.2))

        # 1D Conv
        model.add(Reshape(((sequence_len-0), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

        model.compile(loss='mse', optimizer='adam')
        plot_model(model, to_file='model.png', show_shapes=True)

        return model