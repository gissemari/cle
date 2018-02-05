import ipdb
import cPickle
import logging
import numpy as np
import os
import sys
import theano
import theano.tensor as T
import time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from cle.cle.graph import TheanoMixin
from cle.cle.utils import secure_pickle_dump, tolist
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams # for theano_rng used in sampling (to plot)
from theano.tensor.raw_random import RandomStreamsBase as RandStrBase
from random import *

logger = logging.getLogger(__name__)


class Extension(object):
    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()


class GradientClipping(Extension):
    def __init__(self, scaler=5, batch_size=1, check_nan=0):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_grad'
        self.scaler = scaler
        self.batch_size = batch_size
        self.check_nan = check_nan

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        grads = mainloop.grads
        g_norm = 0.

        for p, g in grads.items():
            g /= T.cast(self.batch_size, dtype=theano.config.floatX)
            grads[p] = g
            g_norm += (g**2).sum()

        if self.check_nan:
            not_finite = T.or_(T.isnan(g_norm), T.isinf(g_norm))

        g_norm = T.sqrt(g_norm)
        scaler = self.scaler / T.maximum(self.scaler, g_norm)

        if self.check_nan:
            for p, g in grads.items():
                grads[p] = T.switch(not_finite, 0.1 * p, g * scaler)
        else:
            for p, g in grads.items():
                grads[p] = g * scaler

        mainloop.grads = grads


class EpochCount(Extension):
    def __init__(self, num_epoch):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_term'
        self.num_epoch = num_epoch

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        if np.mod(mainloop.trainlog.epoch_seen, self.num_epoch) == 0:
            mainloop.endloop = 1


class Monitoring(Extension, TheanoMixin):
    def __init__(self, freq, ddout=None, indexSep = 22, indexDDoutPlot = None,
                   instancesPlot = None, data=None, savedFolder = None,monitor_fn=None,
                 obj_monitor_fn=None, obj_monitor_ch=[], explosion_limit=1e8):
        """
        obj_monitor_fn :
            Python function, a function adapted to the mean of main objective,
            e.g., perplexity
        .. todo::

            WRITEME
        """
        self.name = 'ext_monitor'
        self.freq = freq
        self.ddout = ddout
        self.data = data
        self.monitor_fn = monitor_fn
        self.obj_monitor_fn = obj_monitor_fn
        self.obj_monitor_ch = obj_monitor_ch
        self.explosion_limit = explosion_limit
        self.indexSep = indexSep
        self.savedFolder = savedFolder
        self.indexDDoutPlot = indexDDoutPlot
        self.instancesPlot = instancesPlot
        self.firstPlot = 1
        #self.lastResults = None


    def monitor_data_based_channels(self, mainloop):
        """
        .. todo::

            WRITEME
        """

        if self.monitor_fn is None:
            inputs = mainloop.inputs
            self.monitor_fn = self.build_theano_graph(inputs, self.ddout)
        count=0
        if self.data is not None:
            data_record = []
            others_record = []
            for data in self.data: #       
                batch_record = []
                others = []
                for batch in data: # data es un iterator - batch es tuple ([batches[0], mask]) this happened n_batchs times
                    #batch[0].shape -> (726, 20, 3)
                    this_out = self.monitor_fn(*batch) # len(this_out) = 20 = batch size
                    batch_record.append(this_out[:self.indexSep]) #indexSep 18
                    others.append(this_out[self.indexSep:]) # 5 batches
                    #y_pred.append(batch[2])
                    
                    #print("here", len(this_out[indexSep:]), len(this_out[indexSep:][0]))
                    #7 batches if 10000 instances uploaded, I guess because valid set does not have more than 1400
                    ### Plot here real batches X
                    if (self.firstPlot==1):
                        oneBatch = np.concatenate(np.squeeze(batch[0]), axis = 0)
                        plt.figure(1)
                        plt.plot(oneBatch)
                        plt.savefig("{}/x_batch-{}".format(self.savedFolder,count))
                        plt.clf()
                        if (len(batch)>2): # Ploting Y
                            plt.figure(2)
                            oneBatch = np.concatenate(np.squeeze(batch[2]), axis = 0)
                            plt.plot(oneBatch)
                            plt.savefig("{}/y-{}".format(self.savedFolder,count))
                            plt.clf()
                        self.firstPlot=0
                    count+=1
                print(count)
                data_record.append(np.asarray(batch_record))
                others_record.append(others)
                #others_record.append(np.asarray(others))
            numSelectedBatch = np.random.randint(count)-1
            epoch = mainloop.trainlog.epoch_seen
            rows = len(self.ddout) - self.indexSep
            f, axorig = plt.subplots(rows, 1, sharex=True)
            for record, data in zip(data_record, self.data):
                strLog = ''
                for i, ch in enumerate(self.ddout):
                    if (i>=self.indexSep): # number of parameters that just need mean to be measured
                        #### PLOTING FOR JUST SERIES. Maybe another FOR for the different batches in different files
                        oneBatch = np.concatenate(others_record[0][numSelectedBatch][i-self.indexSep], axis = 0)#.reshape((y_real1[0].shape[0]*len(y_real1[0:2])*y_real1[0].shape[1],-1))
                        axorig[i-self.indexSep].plot(oneBatch)
                        axorig[i-self.indexSep].set_title('{}'.format(ch.name))
                        ####
                    else:
                        this_mean = record[:, i].mean() #mean among the batches
                        #if (ch.name[0:3] = 'mse' ): # not necessary because this is .mean()
                        #    this_mean = record[:, i].sum() * (1/count)
                        if this_mean is np.nan:
                            raise ValueError("NaN occured in output.")
                        strLog +="{}: {} ".format(ch.name, this_mean)
                        #logger.info(" %s_%s: %f " % (data.name, ch.name, this_mean))
                        #print(record[:, i].shape[0],record[:, i].shape[1])
                        if this_mean > self.explosion_limit:
                            raise ValueError('explosion')

                        #ch_name = "%s_%s" % (data.name, ch.name)
                        ch_name = "%s" % (ch.name)
                        mainloop.trainlog.monitor[ch_name].append(this_mean)

                        if i < len(self.obj_monitor_ch) and self.obj_monitor_fn is not None:
                            obj_monitor_val = self.obj_monitor_fn(this_mean)
                            ch_name = "%s_%s" % (data.name, self.obj_monitor_ch[i])
                            logger.info(" %s: %f" % (ch_name, obj_monitor_val))
                            mainloop.trainlog.monitor[ch_name].append(obj_monitor_val)
                print(strLog)
                f.subplots_adjust(top=0.92, bottom=0.05, left=0.10, right=0.95, hspace=0.3, wspace=0.3)
                plt.savefig("{}/all-{}-batch_e{}".format(self.savedFolder,numSelectedBatch,epoch), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)
                plt.clf()
            '''
            for record, data in zip(others_record, self.data):
                numBatch=0
                #y_real1 = []
                x_real1 = []
                restOfRecords = {ch[1].name:[] for i, ch in enumerate(self.indexDDoutPlot)}

                for batch in data:
                    #y_real1.append(batch[2])
                    x_real1.append(batch[0])
                    for i, ch in enumerate(self.indexDDoutPlot):
                        restOfRecords[ch[1].name].append(record[numBatch][ch[0]])

                    if (numBatch==0):
                        #y_real = batch[2]#0-batch_x,1-mask_x, 2-labels, 3-mask_label
                        x_real = batch[0]
                        numfig = 1

                        
                        if (self.firstPlot ==1):
                            f, axorig = plt.subplots(2, cols, sharex=True)
                            for j, instance in enumerate(self.instancesPlot):

                                ################# DISAGGREGATION
                                #plt.figure(numfig)
                                #plt.subplot(212)
                                if (len(y_real.shape)==3):
                                    axorig[0,j].plot(y_real[:,instance,:])
                                else:
                                    axorig[0,j].plot(y_real[:,instance])

                                axorig[0,j].set_title('Y-original-{}'.format(instance))
                                #plt.savefig("{}/DisagReal_{}".format(self.savedFolder,instance))
                                #plt.clf()
                                ################ X ORIGINAL
                                axorig[1,j].plot(x_real[:,instance,:])
                                axorig[1,j].set_title('X-original-{}'.format(instance))
                            plt.savefig("{}/originalInstances".format(self.savedFolder))
                            plt.clf()
                            
                                #plt.figure(figsize=(20,50)) # makes just the two in the botton appear
                                #plt.rcParams["figure.figsize"] = (50,100)
                        plt.rcParams['figure.figsize'] = [20, 20]
                        plt.rcParams['font.size'] = 20
                        f, axarr = plt.subplots(rows, cols, sharex=True)

                        for j, instance in enumerate(self.instancesPlot):    
                            for i, ch in enumerate(self.indexDDoutPlot):
                                 # number of parameters that just need mean to be measured
                                this_var_batch = record[numBatch][ch[0]] # record[numBatch,i-18]
                                #22-28: binary_temp, corr_temp, theta_mu_temp, theta_sig_temp, s_temp, z_1_temp, coeff_temp
                                # int(np.random.RandomState(np.random.randint(1024)))#RandomStreams().uniform()#RandStrBase().random_integers(low=0, high=this_var_batch.shape[1]) #batchsize
                                fig  = 1+i*cols + j
                                #print(fig)
                                axarr[i,j].plot(this_var_batch[:,instance,:])
                                axarr[i,j].set_title(ch[1].name)
                        f.subplots_adjust(top=0.92, bottom=0.05, left=0.10, right=0.95, hspace=0.3, wspace=0.3)
                        plt.savefig("{}/allTogetherInstances_{}".format(self.savedFolder,epoch), bbox_inches='tight')#self.savedFolder+'/'+ch.name+str(numfig)
                        plt.clf()
                        numBatch+=1
                    #print('Prior ', len(others_record), len(others_record[0]),len(others_record[0][0]) )#, len(others_record[1])
                #y_allBatch = np.concatenate(y_real1[0:2], axis = 0).reshape((y_real1[0].shape[0]*len(y_real1[0:2])*y_real1[0].shape[1],-1))
                x_allBatch = np.concatenate(x_real1[0:2], axis = 0).reshape((x_real1[0].shape[0]*len(x_real1[0:2])*x_real1[0].shape[1],-1))
                if (self.firstPlot == 1):
                    f, axorig = plt.subplots(2, 1, sharex=True)
                    axorig[0].plot(y_allBatch)
                    axorig[0].set_title('Y-original')
                    axorig[1].plot(x_allBatch)
                    axorig[1].set_title('X-original')
                    plt.savefig("{}/XY_{}".format(self.savedFolder,epoch))#self.savedFolder+'/'+ch.name+str(numfig)
                    plt.clf()
                    self.firstPlot = 0
                f, axarr = plt.subplots(rows,1, sharex=True)
                for i, ch in enumerate(self.indexDDoutPlot):
                    aux0 = restOfRecords[ch[1].name]
                    aux = np.concatenate(aux0, axis = 0).reshape((aux0[0].shape[0]*len(aux0)*aux0[0].shape[1],-1))#T.concatenate(restOfRecords[ch[1].name], axis = 1)
                    axarr[i].plot(aux)
                    axarr[i].set_title(ch[1].name)
                plt.savefig("{}/allTogetherAllSet_{}".format(self.savedFolder,epoch))#self.savedFolder+'/'+ch.name+str(numfig)
                plt.clf()
            '''
        else:
            pass

    def monitor_data_plot(self, mainloop):
        if self.monitor_fn is None:
            inputs = mainloop.inputs
            self.monitor_fn = self.build_theano_graph(inputs, self.ddout)

        if self.data is not None:
            data_record = []
            others_record = []
            #y_record = []
            for data in self.data: #       
                batch_record = []
                count=0
                for batch in data: # data es un iterator - batch es tuple ([batches[0], mask]) this happened n_batchs times
                    #batch[0].shape -> (726, 20, 3)
                    this_out = self.monitor_fn(*batch) # len(this_out) = 20 = batch size
                    batch_record.append(this_out[:self.indexSep]) #indexSep 18                    
                    count+=1
                print("Batches in val: {}".format(count))
                data_record.append(batch_record)

            self.lastResults = data_record
            cols=len(self.instancesPlot)
            rows = len(self.indexDDoutPlot)
            numfig = 1
            epoch = mainloop.trainlog.epoch_seen
            for record, data in zip(data_record, self.data):
                numBatch=0
                for batch in data:
                    if (numBatch==0):
                        y_real = batch[2]#0-batch_x,1-mask_x, 2-labels, 3-mask_label
                        x_real = batch[0]
                        numfig = 1
                        if (self.firstPlot ==1):
                            f, axorig = plt.subplots(2, cols, sharex=True)
                            for j, instance in enumerate(self.instancesPlot):

                                axorig[0,j].plot(y_real[:,instance,:])
                                axorig[0,j].set_title('Y-original-{}'.format(instance))

                                axorig[1,j].plot(x_real[:,instance,:])
                                axorig[1,j].set_title('X-original-{}'.format(instance))
                            plt.savefig("{}/original".format(self.savedFolder))
                            plt.clf()
                            self.firstPlot = 0

                        plt.rcParams['figure.figsize'] = [20, 20]
                        plt.rcParams['font.size'] = 20
                        f, axarr = plt.subplots(rows, cols, sharex=True)

                        for j, instance in enumerate(self.instancesPlot):    
                            for i, ch in enumerate(self.indexDDoutPlot):
                                 # number of parameters that just need mean to be measured
                                this_var_batch = record[numBatch][ch[0]] # 0: index - 1: name
                                fig  = 1+i*cols + j
                                if (ch[1].name=='disaggregation'):
                                    continue
                                axarr[i,j].plot(this_var_batch[:,instance,:])
                                axarr[i,j].set_title(ch[1].name)
                        f.subplots_adjust(top=0.92, bottom=0.05, left=0.10, right=0.95, hspace=0.3, wspace=0.3)
                        plt.savefig("{}/allTogether_{}".format(self.savedFolder,epoch))#self.savedFolder+'/'+ch.name+str(numfig)
                        plt.clf()
                        numBatch+=1
        else:
            pass


    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        log = mainloop.trainlog
        if np.mod(log.batch_seen, self.freq) == 0 or mainloop.endloop:
            srt = max(0, log.batch_seen - self.freq)
            end = max(1, log.batch_seen)
            t = np.asarray(log.monitor['time'])[srt: end].sum()
            logger.info("")
            logger.info(" Monitoring step")
            logger.info(" Elapsed time: %f epochs %d batches seen %d" % (t, log.epoch_seen, log.batch_seen))
            #logger.info(" Epochs  seen: %d" % )
            #logger.info(" Batches seen: %d" % )
            logger.info(" Optimization parameters")
            logger.info(" .......................")
            mainloop.optimizer.monitor()
            logger.info(" ------------------")
            
            output_channel = [out.name for out in mainloop.outputs]
            logger.info(" Forward-prop based - len output-channels %d" % len(output_channel))
            if log.batch_seen == 0:
                logger.info(" initial_monitoring")
            else:
                for i, out in enumerate(output_channel):
                    this_mean = np.asarray(log.monitor['update'])[srt: end, i].mean() # just this batch metrics?
                    if this_mean is np.nan:
                        raise ValueError("NaN occured in output.")
                    logger.info(" this_batch_%s: %f" % (out, this_mean))

            this_t0 = time.time()
            self.monitor_data_based_channels(mainloop) # this is done for all batches
            mt = time.time() - this_t0
            logger.info(" Elapsed time for monitoring all batches: %f" % mt)
            mainloop.trainlog.monitor['monitoring_time'] = mt


class Picklize(Extension):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, freq, path, force_save_freq=1e15):
        self.name = 'ext_save'
        self.freq = freq
        self.force_save_freq = force_save_freq
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def exe(self, mainloop):
        """
        Pickle the mainloop
        """
        if np.mod(mainloop.trainlog.batch_seen, self.freq) == 0 or mainloop.endloop:
            pkl_path = mainloop.name + '.pkl'
            path = os.path.join(self.path, pkl_path)
            logger.info(" Saving model to: %s" % path)

            try:
                import sys
                sys.setrecursionlimit(50000)
                f = open(path, 'wb')
                cPickle.dump(mainloop, f, -1)
                f.close()
                #secure_pickle_dump(mainloop, path)
            except Exception:
                raise

        if np.mod(mainloop.trainlog.batch_seen, self.force_save_freq) == 0 and\
                mainloop.trainlog.batch_seen != 0:
            force_pkl_path = mainloop.name + '_' +\
                             str(mainloop.trainlog.batch_seen) +\
                             'updates.pkl'
            force_path = os.path.join(self.path, force_pkl_path)
            logger.info(" Saving model to: %s" % force_path)

            try:
                import sys
                sys.setrecursionlimit(50000)
                f = open(force_path, 'wb')
                cPickle.dump(mainloop, f, -1)
                f.close()
                #secure_pickle_dump(mainloop, path)
            except Exception:
                raise


class EarlyStopping(Extension):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, path, channel=None, freq=1, force_save_freq=None,
                 minimize=1):
        self.name = 'ext_save'
        self.freq = freq
        self.force_save_freq = force_save_freq

        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path
        self.best = sys.float_info.max
        self.minimize_ = minimize

        if not self.minimize_:
            self.best *= -1

        self.channel = channel

        if self.channel is None:
            self.channel = 'valid_nll'
            #raise AttributeError("channel is required for early stopping.")

    def exe(self, mainloop):
        """
        Pickle the mainloop
        """
        if len(mainloop.trainlog.monitor['update']) > 0:
            if np.mod(mainloop.trainlog.batch_seen, self.freq) == 0 or mainloop.endloop:
                token = 0

                if self.minimize_:
                    if mainloop.trainlog.monitor[self.channel][-1] < self.best:
                        token = 1
                else:
                    if mainloop.trainlog.monitor[self.channel][-1] > self.best:
                        token = 1

                if token:
                    self.best = mainloop.trainlog.monitor[self.channel][-1]
                    pkl_path = mainloop.name + '_best.pkl'
                    path = os.path.join(self.path, pkl_path)
                    logger.info(" Saving best model to: %s" % path)

                    try:
                        import sys
                        sys.setrecursionlimit(50000)
                        f = open(path, 'wb')
                        cPickle.dump(mainloop, f, -1)
                        f.close()
                        #secure_pickle_dump(mainloop, path)
                    except Exception:
                        raise

                    if self.force_save_freq is not None:
                        this_scaler = (mainloop.trainlog.batch_seen /
                                      self.force_save_freq)
                        this_number = self.force_save_freq * (this_scaler + 1)
                        force_pkl_path = mainloop.name + '_best_before_' +\
                                         str(this_number) +\
                                         'updates.pkl'
                        force_path = os.path.join(self.path, force_pkl_path)
                        logger.info(" Saving best model to: %s" % force_path)

                        try:
                            import sys
                            sys.setrecursionlimit(50000)
                            f = open(force_path, 'wb')
                            cPickle.dump(mainloop, f, -1)
                            f.close()
                            #secure_pickle_dump(mainloop, path)
                        except Exception:
                            raise


class WeightDecay(Extension):
    def __init__(self, lambd=0.0002, keys=['W']):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_regularize_pre_grad'
        self.lambd = lambd
        self.keys = tolist(keys)

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        for k, p in mainloop.model.params.items():
            for key in self.keys:
                if key in k:
                    mainloop.cost += self.lambd * T.sqr(p).sum()


class WeightNorm(Extension):
    def __init__(self, is_vector=1, weight_norm=1.9365, keys=['W'], waivers=[]):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_regularize_post_grad'
        self.weight_norm = weight_norm
        self.keys = tolist(keys)
        self.waivers = tolist(waivers)
        self.is_vector = is_vector

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        for k, p in mainloop.updates.items():
            for key in self.keys:
                if key in str(k):
                    token = 1

                    for waiver in self.waivers:
                        if waiver in str(k):
                            token = 0

                    if token:
                        updated_param = mainloop.updates[k]

                        if self.is_vector:
                            col_norms = T.sqrt(T.sqr(updated_param).sum(axis=0))
                            desired_norms = T.clip(col_norms, 0, self.weight_norm)
                            ratio = (desired_norms / (1e-7 + col_norms))
                            mainloop.updates[k] = updated_param * ratio
                        else:
                            norm = T.sqrt(T.sqr(updated_param).sum())
                            desired_norm = T.clip(norm, 0, self.weight_norm)
                            ratio = (desired_norm / (1e-7 + norm))
                            mainloop.updates[k] = updated_param * ratio


class LrLinearDecay(Extension):
    def __init__(self, start, end, decay_factor):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_schedule'
        assert start > 0
        assert end > start
        self.start = start
        self.end = end
        self.decay_factor = decay_factor
        self.count = 0

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        if self.count == 0:
            self.base_lr = mainloop.optimizer.lr.get_value()
            self.step = ((self.base_lr - self.base_lr * self.decay_factor) /
                         (self.end - self.start + 1))

        self.count += 1

        if self.count >= self.start:
            if self.count < self.end:
                new_lr = self.base_lr - self.step * (self.count - self.start + 1)
            else:
                new_lr = self.base_lr * self.decay_factor
        else:
            new_lr = self.base_lr

        assert new_lr > 0
        new_lr = np.cast[theano.config.floatX](new_lr)
        mainloop.optimizer.lr.set_value(new_lr)


class LrExponentialDecay(Extension):
    def __init__(self, decay_factor, min_lr):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_schedule'
        self.count = 0
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.min_ = False

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        if self.count == 0:
            self.base_lr = mainloop.optimizer.lr.get_value()

        self.count += 1

        if not self.min_:
            new_lr = self.base_lr / (self.decay_factor ** self.count)
            if new_lr <= self.min_lr:
                self.min_ = True
                new_lr = self.min_lr
        else:
            new_lr = self.min_lr

        new_lr = np.cast[theano.config.floatX](new_lr)
        mainloop.optimizer.lr.set_value(new_lr)
