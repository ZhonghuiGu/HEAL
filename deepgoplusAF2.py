import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import math
import torch
import os
from utils import load_GO_annot
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Conv1D, Flatten, Concatenate,
    MaxPooling1D, Dropout, RepeatVector, Layer
)
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import pickle as pkl
#from utils import Ontology, FUNC_DICT
#from aminoacids import to_ngrams, to_onehot, MAXLEN

logging.basicConfig(level=logging.INFO)

MAXLEN = 1000

# config = tf.ConfigProto(allow_soft_placement=True) not required if 
tf.config.set_soft_device_placement(True)


# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# K.set_session(session)

@ck.command()
@ck.option(
    '--model-file', '-mf', default='model/deepgoplusAF2_model.h5',
    help='DeepGOPlus model')
@ck.option(
    '--out-file', '-o', default='model/deepgoplusAF2_AF2test_predictions.pkl',
    help='Result file with predictions for test set')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=100,
    help='Training epochs')
@ck.option(
    '--logger-file', '-lf', default='model/deepgoplusAF2_training.csv',
    help='Batch size')
@ck.option(
    '--device', '-d', default='gpu:0',
    help='Prediction threshold')
@ck.option(
    '--params-index', '-pi', default=-1,
    help='Definition mapping file')
def main(model_file,
         out_file, batch_size, epochs, logger_file,
         device, params_index, load=False):
    params = {
        'max_kernel': 129,
        'initializer': 'glorot_normal',
        'dense_depth': 0,
        'nb_filters': 512,
        'optimizer': Adam(lr=3e-4),
        'loss': 'binary_crossentropy'
    }
    # SLURM JOB ARRAY INDEX
    pi = params_index = -1
    if params_index != -1:
        kernels = [33, 65, 129, 257, 513]
        dense_depths = [0, 1, 2]
        nb_filters = [32, 64, 128, 256, 512]
        params['max_kernel'] = kernels[pi % 5]
        pi //= 5
        params['dense_depth'] = dense_depths[pi % 3]
        pi //= 3
        params['nb_filters'] = nb_filters[pi % 5]
        pi //= 5
#         out_file = f'model/predictions_{params_index}.pkl'
        logger_file = f'model/training_{params_index}.csv'
        model_file = f'model/model_{params_index}.h5'
    print('Params:', params)
    
    
    with tf.device('/' + device):
        
        test_generator = DFGenerator('AF2test', batch_size)
        test_steps = int(math.ceil(test_generator.size / batch_size))
        nb_classes = test_generator.nb_classes
        if load:
            logging.info('Loading pretrained model')
            model = load_model(model_file)
        else:
            logging.info('Creating a new model')
            model = create_model(nb_classes, params)
            
            checkpointer = ModelCheckpoint(
                filepath=model_file,
                verbose=1, save_best_only=True)
            earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
            logger = CSVLogger(logger_file)

            train_generator = DFGenerator('train', batch_size)
            valid_generator = DFGenerator('val', batch_size)

            valid_steps = int(math.ceil(valid_generator.size / batch_size))
            train_steps = int(math.ceil(train_generator.size / batch_size))
            
            logging.info('Starting training the model')
            logging.info("Training data size: %d" % train_generator.size)
            logging.info("Validation data size: %d" % valid_generator.size)
    
            
            model.summary()
            model.fit(
                train_generator,
                steps_per_epoch=train_steps,
                epochs=epochs,
                validation_data=valid_generator,
                validation_steps=valid_steps,
                max_queue_size=batch_size,
                workers=12,
                callbacks=[logger, checkpointer, earlystopper])
            
            logging.info('Loading best model')
            model = load_model(model_file)

    
        logging.info('Evaluating model')
        loss = model.evaluate(test_generator, steps=test_steps)
        logging.info('Test loss %f' % loss)
        logging.info('Predicting')
        test_generator.reset()
        preds = model.predict(test_generator, steps=test_steps)
        
        # valid_steps = int(math.ceil(len(valid_df) / batch_size))
        # valid_generator = DFGenerator(valid_df, terms_dict,
        #                               nb_classes, batch_size)
        # logging.info('Predicting')
        # valid_generator.reset()
        # preds = model.predict_generator(valid_generator, steps=valid_steps)
        # valid_df.reset_index()
        # valid_df['preds'] = list(preds)
        # train_df.to_pickle('data/train_data_train.pkl')
        # valid_df.to_pickle('data/train_data_valid.pkl')
        with open(out_file,'wb') as f:
            pkl.dump(preds, f)
        

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def create_model(nb_classes, params):
    inp_hot = Input(shape=(MAXLEN, 21), dtype=np.float32)
    
    kernels = range(8, params['max_kernel'], 8)
    nets = []
    for i in range(len(kernels)):
        conv = Conv1D(
            filters=params['nb_filters'],
            kernel_size=kernels[i],
            padding='valid',
            name='conv_' + str(i),
            kernel_initializer=params['initializer'])(inp_hot)
        print(conv.get_shape())
        pool = MaxPooling1D(
            pool_size=MAXLEN - kernels[i] + 1, name='pool_' + str(i))(conv)
        flat = Flatten(name='flat_' + str(i))(pool)
        nets.append(flat)

    net = Concatenate(axis=1)(nets)
    for i in range(params['dense_depth']):
        net = Dense(nb_classes, activation='relu', name='dense_' + str(i))(net)
    net = Dense(nb_classes, activation='sigmoid', name='dense_out')(net)
    model = Model(inputs=inp_hot, outputs=net)
    model.summary()
    model.compile(
        optimizer=params['optimizer'],
        loss=params['loss'])
    logging.info('Compilation finished')

    return model

class DFGenerator(Sequence):

    def __init__(self, set_type, batch_size):
        # set_type in ['train', 'val','test']
        self.start = 0

        self.batch_size = batch_size 
        #if set_type != 'AF2test':
        prot2annot, goterms, gonames, counts = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")
        #else:
        prot2annot1, goterms1, gonames1, counts1 = load_GO_annot("data/nrSwiss-Model-GO_annot.tsv")
        
        self.processed_dir = 'data/processed'
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            print(f'using processed directory: {self.processed_dir}')
        
        #goterm_dataset = GoTermDataset(set_type, 'mf', True)

        self.data_path = os.path.join(self.processed_dir, f"{set_type}_graph.pt")

        self.graph_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_graph.pt")) 
        if set_type != 'AF2test':
            self.graph_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_graph.pt")) + torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_graph.pt"))
            self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]+ torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
        else:
            self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))[f"test_pdbch"]

        self.size = len(self.pdbch_list)
        
        self.y_true = []
        for pdb_c in self.pdbch_list:
            if pdb_c in prot2annot.keys():
                self.y_true.append( np.concatenate([prot2annot[pdb_c][task] for task in ['mf','bp','cc']], -1) )
            else:
                self.y_true.append( np.concatenate([prot2annot1[pdb_c][task] for task in ['mf','bp','cc']], -1) )
        self.y_true = np.stack(self.y_true)
        self.nb_classes = self.y_true.shape[-1]
        
        self.MAXLEN = 1000

    ### copied from deepgopp    
    def __len__(self):                                                                                                                   
        return np.ceil(self.size / float(self.batch_size)).astype(np.int32)   

    def __getitem__(self, idx):                                                                                                          
        batch_index = np.arange(                                                                                                         
            idx * self.batch_size, min(self.size, (idx + 1) * self.batch_size))                                                          
        
        data_onehots = []
        for idx in batch_index:
            graph = self.graph_list[idx]
            data_onehot = torch.nn.functional.one_hot(graph.native_x.long(), 21).numpy()
            data_onehot = np.concatenate([data_onehot, np.zeros([self.MAXLEN - data_onehot.shape[0], 21]).astype(np.int64)], 0)
            data_onehots.append(data_onehot)
        data_onehots = np.stack(data_onehots).astype(np.int32)
         
        labels = self.y_true[batch_index].astype(np.int32)
        self.start += self.batch_size
        #print(data_onehot, labels)
        return (data_onehots, labels)
    ###################

    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            data_onehots = []
            for idx in batch_index:
                graph = self.graph_list[idx]
                data_onehot = torch.nn.functional.one_hot(graph.native_x.long(), 21).numpy()
                data_onehot = np.concatenate([data_onehot, np.zeros([self.MAXLEN - data_onehot.shape[0], 21]).astype(np.int64)], 0)
                data_onehots.append(data_onehot)
            data_onehots = np.stack(data_onehots).astype(np.int32)

            labels = self.y_true[batch_index].astype(np.int32)
            self.start += self.batch_size
            #print(data_onehot, labels)
            return (data_onehots, labels)
        else:
            self.reset()
            return self.next()

if __name__ == '__main__':
    main()
