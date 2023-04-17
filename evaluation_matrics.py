# This script is written based on DeepFRI evaluation.py https://github.com/flatironinstitute/DeepFRI

import csv
import pickle
import obonet
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from sklearn.metrics import average_precision_score as aupr

import seaborn as sns
from matplotlib import pyplot as plt
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('font', family='arial')

# go.obo
go_graph = obonet.read_obo(open("data/go-basic.obo", 'r'))


def bootstrap(Y_true, Y_pred):
    n = Y_true.shape[0]
    idx = np.random.choice(n, n)

    return Y_true[idx], Y_pred[idx]


def load_test_prots(fn):
    proteins = []
    seqid_mtrx = []
    with open(fn, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            inds = row[1:]
            inds = np.asarray([int(i) for i in inds]).reshape(1, len(inds))
            proteins.append(row[0])
            seqid_mtrx.append(inds)

    return np.asarray(proteins), np.concatenate(seqid_mtrx, axis=0)


def load_go2ic_mapping(fn):
    goterm2ic = {}
    fRead = open(fn, 'r')
    for line in fRead:
        goterm, ic = line.strip().split()
        goterm2ic[goterm] = float(ic)
    fRead.close()

    return goterm2ic


def propagate_go_preds(Y_hat, goterms):
    go2id = {go: ii for ii, go in enumerate(goterms)}
    for goterm in goterms:
        if goterm in go_graph:
            parents = set(goterms).intersection(nx.descendants(go_graph,
                                                               goterm))
            for parent in parents:
                Y_hat[:, go2id[parent]] = np.maximum(Y_hat[:, go2id[goterm]],
                                                     Y_hat[:, go2id[parent]])

    return Y_hat


def propagate_ec_preds(Y_hat, goterms):
    go2id = {go: ii for ii, go in enumerate(goterms)}
    for goterm in goterms:
        if goterm.find('-') == -1:
            parent = goterm.split('.')
            parent[-1] = '-'
            parent = ".".join(parent)
            if parent in go2id:
                Y_hat[:, go2id[parent]] = np.maximum(Y_hat[:, go2id[goterm]],
                                                     Y_hat[:, go2id[parent]])

    return Y_hat

''' helper functions follow '''
def normalizedSemanticDistance(Ytrue, Ypred, termIC, avg=False, returnRuMi = False):
    '''
    evaluate a set of protein predictions using normalized semantic distance
    value of 0 means perfect predictions, larger values denote worse predictions,
    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, predicted binary label ndarray (not compressed). Must have hard predictions (0 or 1, not posterior probabilities)
        termIC: output of ic function above
    OUTPUT:
        depending on returnRuMi and avg. To get the average sd over all proteins in a batch/dataset
        use avg = True and returnRuMi = False
        To get result per protein, use avg = False
    '''

    ru = normalizedRemainingUncertainty(Ytrue, Ypred, termIC, False)
    mi = normalizedMisInformation(Ytrue, Ypred, termIC, False)
    sd = np.sqrt(ru ** 2 + mi ** 2)

    if avg:
        ru = np.mean(ru)
        mi = np.mean(mi)
        sd = np.sqrt(ru ** 2 + mi ** 2)

    if not returnRuMi:
        return sd

    return [ru, mi, sd]

def normalizedRemainingUncertainty(Ytrue, Ypred, termIC, avg=False):
    num =  np.logical_and(Ytrue == 1, Ypred == 0).astype(float).dot(termIC)
    denom =  np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nru = num / denom

    if avg:
        nru = np.mean(nru)

    return nru

def normalizedMisInformation(Ytrue, Ypred, termIC, avg=False):
    num =  np.logical_and(Ytrue == 0, Ypred == 1).astype(float).dot(termIC)
    denom =  np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nmi = num / denom

    if avg:
        nmi = np.mean(nmi)

    return nmi

### prepare the Information content dict beforehand
with open("data/ic_count.pkl",'rb') as f:
    ic_count = pkl.load(f)
ic_count['bp'] = np.where(ic_count['bp']==0, 1, ic_count['bp'])
ic_count['mf'] = np.where(ic_count['mf']==0, 1, ic_count['mf'])
ic_count['cc'] = np.where(ic_count['cc']==0, 1, ic_count['cc'])
train_ic = {}
train_ic['bp'] = -np.log2(ic_count['bp'] / 69709)
train_ic['mf'] = -np.log2(ic_count['mf'] / 69709)
train_ic['cc'] = -np.log2(ic_count['cc'] / 69709)

class Method(object):
    def __init__(self, method_name, pckl_fn):
        annot = pickle.load(open(pckl_fn, 'rb'))
        self.Y_true = annot['Y_true']
        self.Y_pred = annot['Y_pred']
        self.goterms = annot['goterms']
        self.gonames = annot['gonames']
        self.proteins = annot['proteins']
        self.ont = annot['ontology']
        self.method_name = method_name
        self._propagate_preds()
        if self.ont == 'ec':
            goidx = [i for i, goterm in enumerate(self.goterms) if goterm.find('-') == -1]
            self.Y_true = self.Y_true[:, goidx]
            self.Y_pred = self.Y_pred[:, goidx]
            self.goterms = [self.goterms[idx] for idx in goidx]
            self.gonames = [self.gonames[idx] for idx in goidx]
        self.termIC = train_ic[self.ont]

    def _propagate_preds(self):
        if self.ont == 'ec':
            self.Y_pred = propagate_ec_preds(self.Y_pred, self.goterms)
        else:
            self.Y_pred = propagate_go_preds(self.Y_pred, self.goterms)

    def _cafa_ec_aupr(self, labels, preds):
        # propagate goterms (take into account goterm specificity)

        # number of test proteins
        n = labels.shape[0]

        goterms = np.asarray(self.goterms)

        prot2goterms = {}
        for i in range(0, n):
            prot2goterms[i] = set(goterms[np.where(labels[i] == 1)[0]])

        # CAFA-like F-max predictions
        F_list = []
        AvgPr_list = []
        AvgRc_list = []
        thresh_list = []

        for t in range(1, 100):
            threshold = t/100.0
            predictions = (preds > threshold).astype(np.int)

            m = 0
            precision = 0.0
            recall = 0.0
            for i in range(0, n):
                pred_gos = set(goterms[np.where(predictions[i] == 1)[0]])
                num_pred = len(pred_gos)
                num_true = len(prot2goterms[i])
                num_overlap = len(prot2goterms[i].intersection(pred_gos))
                if num_pred > 0:
                    m += 1
                    precision += float(num_overlap)/num_pred
                if num_true > 0:
                    recall += float(num_overlap)/num_true

            if m > 0:
                AvgPr = precision/m
                AvgRc = recall/n

                if AvgPr + AvgRc > 0:
                    F_score = 2*(AvgPr*AvgRc)/(AvgPr + AvgRc)
                    # record in list
                    F_list.append(F_score)
                    AvgPr_list.append(AvgPr)
                    AvgRc_list.append(AvgRc)
                    thresh_list.append(threshold)

        F_list = np.asarray(F_list)
        AvgPr_list = np.asarray(AvgPr_list)
        AvgRc_list = np.asarray(AvgRc_list)
        thresh_list = np.asarray(thresh_list)

        return AvgRc_list, AvgPr_list, F_list, thresh_list

    def _cafa_go_aupr(self, labels, preds):
        # propagate goterms (take into account goterm specificity)

        # number of test proteins
        n = labels.shape[0]

        goterms = np.asarray(self.goterms)
        ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}

        prot2goterms = {}
        for i in range(0, n):
            all_gos = set()
            for goterm in goterms[np.where(labels[i] == 1)[0]]:
                all_gos = all_gos.union(nx.descendants(go_graph, goterm))
                all_gos.add(goterm)
            all_gos.discard(ont2root[self.ont])
            prot2goterms[i] = all_gos

        # CAFA-like F-max predictions
        F_list = []
        AvgPr_list = []
        AvgRc_list = []
        thresh_list = []

        for t in range(1, 100):
            threshold = t/100.0
            predictions = (preds > threshold).astype(np.int)

            m = 0
            precision = 0.0
            recall = 0.0
            for i in range(0, n):
                pred_gos = set()
                for goterm in goterms[np.where(predictions[i] == 1)[0]]:
                    pred_gos = pred_gos.union(nx.descendants(go_graph,
                                                             goterm))
                    pred_gos.add(goterm)
                pred_gos.discard(ont2root[self.ont])

                num_pred = len(pred_gos)
                num_true = len(prot2goterms[i])
                num_overlap = len(prot2goterms[i].intersection(pred_gos))
                if num_pred > 0 and num_true > 0:
                    m += 1
                    precision += float(num_overlap)/num_pred
                    recall += float(num_overlap)/num_true

            if m > 0:
                AvgPr = precision/m
                AvgRc = recall/n

                if AvgPr + AvgRc > 0:
                    F_score = 2*(AvgPr*AvgRc)/(AvgPr + AvgRc)
                    # record in list
                    F_list.append(F_score)
                    AvgPr_list.append(AvgPr)
                    AvgRc_list.append(AvgRc)
                    thresh_list.append(threshold)

        F_list = np.asarray(F_list)
        AvgPr_list = np.asarray(AvgPr_list)
        AvgRc_list = np.asarray(AvgRc_list)
        thresh_list = np.asarray(thresh_list)

        return AvgRc_list, AvgPr_list, F_list, thresh_list

    def _function_centric_aupr(self, keep_pidx=None, keep_goidx=None):
        """ Compute functon-centric AUPR """
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred

        if keep_goidx is not None:
            tmp = []
            for goidx in keep_goidx:
                if Y_true[:, goidx].sum() > 0:
                    tmp.append(goidx)
            keep_goidx = tmp
        else:
            keep_goidx = np.where(Y_true.sum(axis=0) > 0)[0]

        print ("### Number of functions =%d" % (len(keep_goidx)))

        Y_true = Y_true[:, keep_goidx]
        Y_pred = Y_pred[:, keep_goidx]

        # if self.method_name.find('FFPred') >= 0:
        #    goidx = np.where(Y_pred.sum(axis=0) > 0)[0]
        #    Y_true = Y_true[:, goidx]
        #    Y_pred = Y_pred[:, goidx]

        # micro average
        micro_aupr = aupr(Y_true, Y_pred, average='micro')
        # macro average
        macro_aupr = aupr(Y_true, Y_pred, average='macro')

        # each function
        aupr_goterms = aupr(Y_true, Y_pred, average=None)

        return micro_aupr, macro_aupr, aupr_goterms

    def _protein_centric_fmax(self, keep_pidx=None):
        """ Compute protein-centric AUPR """
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred

        # compute recall/precision
        if self.ont in {'mf', 'bp', 'cc'}:
            Recall, Precision, Fscore, thresholds = self._cafa_go_aupr(Y_true,
                                                                       Y_pred)
        else:
            Recall, Precision, Fscore, thresholds = self._cafa_ec_aupr(Y_true,
                                                                       Y_pred)
        return Fscore, Recall, Precision, thresholds

    def fmax(self, keep_pidx):
        fscore, _, _, _ = self._protein_centric_fmax(keep_pidx=keep_pidx)

        return max(fscore)

    def macro_aupr(self, keep_pidx=None, keep_goidx=None):
        _, macro_aupr, _ = self._function_centric_aupr(keep_pidx=keep_pidx, keep_goidx=keep_goidx)
        return macro_aupr

    def smin(self, keep_pidx=None):
        '''
        get the minimum normalized semantic distance
        INPUTS:
            Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
            Ypred : Nproteins x Ngoterms, posterior probabilities (not compressed, in range 0-1).
            termIC: output of ic function above
            nrThresholds: the number of thresholds to check.
        OUTPUT:
            the minimum nsd that was achieved at the evaluated thresholds
        '''
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred
        
        nrThresholds = 100
        thresholds = np.linspace(0.0, 1.0, nrThresholds)
        ss = np.zeros(thresholds.shape)

        for i, t in enumerate(thresholds):
            ss[i] = normalizedSemanticDistance(Y_true, (Y_pred >=t).astype(int), self.termIC, avg=True, returnRuMi=False)

        return np.min(ss)
