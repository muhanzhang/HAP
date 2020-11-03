#################################################################
# Code written by Edward Choi (mp2893@gatech.edu)
# For bug report, please contact author using the email address
#################################################################

import sys, random, time, argparse
from collections import OrderedDict
import cPickle as pickle
import numpy as np
import pdb
from shutil import copy
import os
from collections import Counter

import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.gradient import disconnected_grad

_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1
config.floatX = 'float32'

def unzip(zipped):
    new_params = OrderedDict()
    for key, value in zipped.iteritems():
        new_params[key] = value.get_value()
    return new_params

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def get_random_weight(dim1, dim2, left=-0.1, right=0.1):
    return np.random.uniform(left, right, (dim1, dim2)).astype(config.floatX)

def load_embedding(options):
    m = np.load(options['embFile'])
    w = (m['w'] + m['w_tilde']) / 2.0
    w = np.float32(w)
    return w

def init_params(options):
    params = OrderedDict()

    np.random.seed(0)
    inputDimSize = options['inputDimSize']
    numAncestors = options['numAncestors']
    embDimSize = options['embDimSize']
    hiddenDimSize = options['hiddenDimSize'] #hidden layer does not need an extra space
    attentionDimSize = options['attentionDimSize']
    numClass = options['numClass']

    params['W_emb'] = get_random_weight(inputDimSize+numAncestors, embDimSize)

    if len(options['embFile']) > 0:
        params['W_emb'] = load_embedding(options)
        options['embDimSize'] = params['W_emb'].shape[1]
        embDimSize = options['embDimSize']

    if not args.no_attention and not args.leaf and not args.rollup:
        params['W_attention'] = get_random_weight(embDimSize*2, attentionDimSize)
        params['b_attention'] = np.zeros(attentionDimSize).astype(config.floatX)
        params['v_attention'] = np.random.uniform(-0.1, 0.1, attentionDimSize).astype(config.floatX)

    params['W_gru'] = get_random_weight(embDimSize, 3*hiddenDimSize)
    params['U_gru'] = get_random_weight(hiddenDimSize, 3*hiddenDimSize)
    params['b_gru'] = np.zeros(3 * hiddenDimSize).astype(config.floatX)

    params['W_output'] = get_random_weight(hiddenDimSize, numClass)
    params['b_output'] = np.zeros(numClass).astype(config.floatX)

    return params

def init_tparams(params):
    tparams = OrderedDict()
    for key, value in params.iteritems():
        #if key == 'W_emb' and args.fixed_embed:
        #    tparams[key] = value
        tparams[key] = theano.shared(value, name=key)
    return tparams

def dropout_layer(state_before, use_noise, trng, prob):
    proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=prob, n=1, dtype=state_before.dtype)), state_before * prob)
    return proj

def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]

def gru_layer(tparams, emb, options):
    hiddenDimSize = options['hiddenDimSize']
    timesteps = emb.shape[0]
    if emb.ndim == 3: n_samples = emb.shape[1]
    else: n_samples = 1

    def stepFn(wx, h, U_gru):
        uh = T.dot(h, U_gru)
        r = T.nnet.sigmoid(_slice(wx, 0, hiddenDimSize) + _slice(uh, 0, hiddenDimSize))
        z = T.nnet.sigmoid(_slice(wx, 1, hiddenDimSize) + _slice(uh, 1, hiddenDimSize))
        h_tilde = T.tanh(_slice(wx, 2, hiddenDimSize) + r * _slice(uh, 2, hiddenDimSize))
        h_new = z * h + ((1. - z) * h_tilde)
        return h_new

    Wx = T.dot(emb, tparams['W_gru']) + tparams['b_gru']
    results, updates = theano.scan(fn=stepFn, sequences=[Wx], outputs_info=T.alloc(numpy_floatX(0.0), n_samples, hiddenDimSize), non_sequences=[tparams['U_gru']], name='gru_layer', n_steps=timesteps)

    return results

def generate_attention(tparams, leaves, ancestors):
    '''
    '''
    attentionInput = T.concatenate([tparams['W_emb'][leaves], tparams['W_emb'][ancestors]], axis=2)
    mlpOutput = T.tanh(T.dot(attentionInput, tparams['W_attention']) + tparams['b_attention']) 
    preAttention = T.dot(mlpOutput, tparams['v_attention'])
    attention = T.nnet.softmax(preAttention)
    return attention
    
def softmax_layer(tparams, emb):
    nom = T.exp(T.dot(emb, tparams['W_output']) + tparams['b_output'])
    denom = nom.sum(axis=2, keepdims=True)
    output = nom / denom
    return output
    
def build_model(tparams, leavesList, ancestorsList, options):
    dropoutRate = options['dropoutRate']
    trng = RandomStreams(123)
    use_noise = theano.shared(numpy_floatX(0.))

    x = T.tensor3('x', dtype=config.floatX)
    y = T.tensor3('y', dtype=config.floatX)
    mask = T.matrix('mask', dtype=config.floatX)
    lengths = T.vector('lengths', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    embList = []
    for leaves, ancestors in zip(leavesList, ancestorsList):
        if args.no_attention:
            tempEmb = (tparams['W_emb'][ancestors]).sum(axis=1)
        else:
            if args.leaf:
                tempEmb = (tparams['W_emb'][leaves[:, 0]])  # only use the leaf embeddings
            elif args.rollup:
                tempEmb = (tparams['W_emb'][ancestors[:, -1]])  # all use their upper-level embeddings
            else:
                tempAttention = generate_attention(tparams, leaves, ancestors)
                tempEmb = (tparams['W_emb'][ancestors] * tempAttention[:,:,None]).sum(axis=1)
        embList.append(tempEmb)

    emb = T.concatenate(embList, axis=0)

    x_emb = T.tanh(T.dot(x, emb))
    hidden = gru_layer(tparams, x_emb, options)
    hidden = dropout_layer(hidden, use_noise, trng, dropoutRate)
    y_hat = softmax_layer(tparams, hidden) * mask[:,:,None]

    logEps = 1e-8
    cross_entropy = -(y * T.log(y_hat + logEps) + (1. - y) * T.log(1. - y_hat + logEps))
    output_loglikelihood = cross_entropy.sum(axis=2).sum(axis=0) / lengths
    cost_noreg = T.mean(output_loglikelihood)

    if options['L2'] > 0.:
        if args.no_attention or args.leaf or args.rollup:
            cost = cost_noreg + options['L2'] * (tparams['W_output']**2).sum()
        else:
            cost = cost_noreg + options['L2'] * ((tparams['W_output']**2).sum() + (tparams['W_attention']**2).sum() + (tparams['v_attention']**2).sum())
    else:
        cost = cost_noreg

    return use_noise, x, y, mask, lengths, cost, cost_noreg, y_hat, emb

'''
def group_y(y, num_bins=4):
    y = [y3 for y1 in y for y2 in y1 for y3 in y2]  # flatten y
    unique, counts = np.unique(y, return_counts=True)
    y_dict = dict(zip(unique, counts))
    sorted_unique = sorted(unique, key=lambda x: y_dict[x])
    percentiles = np.linspace(0, 1, num_bins+1)[1:-1]
    cuts = np.ceil(percentiles * len(unique)).astype(int)
    y_grouped = np.split(np.array(sorted_unique), cuts)
    print('Label frequencies in each group:')
    print([sum([y_dict[x] for x in y]) for y in y_grouped])
    return y_grouped


'''
def group_y(y, num_bins=4):
    y = [y3 for y1 in y for y2 in y1 for y3 in y2]  # flatten y
    unique, counts = np.unique(y, return_counts=True)
    total_counts = counts.sum()
    percentiles = np.linspace(0, 1, num_bins+1)[1:]
    cuts = np.ceil(percentiles * total_counts)
    y_dict = dict(zip(unique, counts))
    sorted_unique = sorted(unique, key=lambda x: y_dict[x])
    count = 0
    y_grouped = []
    cur_group = []
    group_id = 0
    for y in sorted_unique:
        cur_group.append(y)
        count += y_dict[y]
        if count > cuts[group_id]:
            y_grouped.append(cur_group)
            cur_group = []
            group_id += 1
    y_grouped.append(cur_group)
    print('Label frequencies in each group:')
    print([sum([y_dict[x] for x in y]) for y in y_grouped])

    return y_grouped


def load_data(seqFile, labelFile, timeFile=''):
    sequences = np.array(pickle.load(open(seqFile, 'rb')))
    labels = np.array(pickle.load(open(labelFile, 'rb')))
    if len(timeFile) > 0:
        times = np.array(pickle.load(open(timeFile, 'rb')))

    np.random.seed(args.seed)
    dataSize = len(labels)
    ind = np.random.permutation(dataSize)
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest+nValid]
    train_indices = ind[nTest+nValid:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]
    train_set_t = None
    test_set_t = None
    valid_set_t = None

    if len(timeFile) > 0:
        train_set_t = times[train_indices]
        test_set_t = times[test_indices]
        valid_set_t = times[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_t)
    valid_set = (valid_set_x, valid_set_y, valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_t)

    y_grouped = group_y(train_set_y)
    args.y_grouped = y_grouped

    return train_set, valid_set, test_set

def adadelta(tparams, grads, x, y, mask, lengths, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y, mask, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    if args.fixed_embed:
        param_up = [(p, p + ud) for (k, p), ud in zip(tparams.items(), updir) if k != 'W_emb']
    else:
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

    return f_grad_shared, f_update

def padMatrix(seqs, labels, options):
    lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, options['inputDimSize'])).astype(config.floatX)
    y = np.zeros((maxlen, n_samples, options['numClass'])).astype(config.floatX)
    mask = np.zeros((maxlen, n_samples)).astype(config.floatX)

    for idx, (seq, lseq) in enumerate(zip(seqs,labels)):
        for xvec, subseq in zip(x[:,idx,:], seq[:-1]): xvec[subseq] = 1.
        for yvec, subseq in zip(y[:,idx,:], lseq[1:]): yvec[subseq] = 1.
        mask[:lengths[idx], idx] = 1.

    lengths = np.array(lengths, dtype=config.floatX)

    return x, y, mask, lengths

def calculate_cost(test_model, dataset, options, k=20):
    batchSize = options['batchSize']
    n_batches = int(np.ceil(float(len(dataset[0])) / float(batchSize)))
    costSum = 0.0
    dataCount = 0
    total_counter = Counter()
    correct_counter = Counter()
    for index in xrange(n_batches):
        batchX = dataset[0][index*batchSize:(index+1)*batchSize]
        batchY = dataset[1][index*batchSize:(index+1)*batchSize]
        x, y, mask, lengths = padMatrix(batchX, batchY, options)
        cost, y_hat, emb = test_model(x, y, mask, lengths)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                true_labels = np.nonzero(y[i, j, :])[0]
                predictions = np.argsort(y_hat[i, j, :])[-k:]
                for l in true_labels:
                    total_counter[l] += 1
                    correct_counter[l] += np.in1d(l, predictions, assume_unique=True).sum()
                #total_labels += len(true_labels)
                #correct_labels += np.in1d(true_labels, predictions, assume_unique=True).sum()
        costSum += cost * len(batchX)
        dataCount += len(batchX)

    y_grouped = args.y_grouped
    n_groups = len(y_grouped)
    total_labels = [0] * n_groups
    correct_labels = [0] * n_groups
    for i, group in enumerate(y_grouped):
        for l in group:
            correct_labels[i] += correct_counter[l]
            total_labels[i] += total_counter[l]

    acc_at_k_grouped = [x/float(y) for x, y in zip(correct_labels, total_labels)]
    acc_at_k = sum(correct_labels) / float(sum(total_labels))
    return costSum / dataCount, acc_at_k, acc_at_k_grouped, emb

def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()

def build_tree(treeFile):
    treeMap = pickle.load(open(treeFile, 'rb'))
    if not treeMap:
        return [], []
    ancestors = np.array(treeMap.values()).astype('int32')
    ancestors = np.concatenate([ancestors[:, :1], ancestors[:, args.max_level-args.level+1:]], 1)
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)
    leaves = np.array(leaves).astype('int32')
    #pdb.set_trace()
    return leaves, ancestors

def train_GRAM(
    seqFile = 'seqFile.txt',
    labelFile = 'labelFile.txt',
    treeFile='tree.txt',
    embFile='embFile.txt',
    p2cFile='',
    outFile='out.txt',
    inputDimSize= 100,
    numAncestors=100,
    embDimSize= 100,
    hiddenDimSize=200,
    attentionDimSize=200,
    max_epochs=100,
    L2=0.,
    numClass=26679,
    batchSize=100,
    dropoutRate=0.5,
    logEps=1e-8,
    verbose=False
):
    options = locals().copy()

    if p2cFile:
        p2c, c2p = pickle.load(open(p2cFile, 'rb'))

    leavesList = []
    ancestorsList = []
    for i in range(args.max_level, 0, -1): # An ICD9 diagnosis code can have at most five ancestors (including the artificial root) when using CCS multi-level grouper. 
        leaves, ancestors = build_tree(treeFile+'.level'+str(i)+'.pk')
        if leaves==[] and ancestors==[]:
            continue
        sharedLeaves = theano.shared(leaves, name='leaves'+str(i))
        sharedAncestors = theano.shared(ancestors, name='ancestors'+str(i))
        leavesList.append(sharedLeaves)
        ancestorsList.append(sharedAncestors)
    
    print 'Building the model ... ',
    params = init_params(options)
    tparams = init_tparams(params)
    if args.test:
        best_model_name = max([f for f in os.listdir(args.out_dir) if f.endswith('.npz')], key=lambda x: int(x.split('.')[1]))
        tempParams = np.load(os.path.join(args.out_dir, best_model_name))
        for key, value in tempParams.iteritems():
            params[key] = value
        tparams = init_tparams(params)
    use_noise, x, y, mask, lengths, cost, cost_noreg, y_hat, emb =  build_model(tparams, leavesList, ancestorsList, options)
    get_cost = theano.function(inputs=[x, y, mask, lengths], outputs=[cost_noreg, y_hat, emb], name='get_cost')
    print 'done!!'
    
    print 'Constructing the optimizer ... ',
    grads = T.grad(cost, wrt=tparams.values())
    f_grad_shared, f_update = adadelta(tparams, grads, x, y, mask, lengths, cost)
    print 'done!!'

    print 'Loading data ... ',
    trainSet, validSet, testSet = load_data(seqFile, labelFile)
    n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
    print 'done!!'

    logFile = outFile + '.log'
    if args.test:
        use_noise.set_value(0.)
        testCost, testAcc, testAccGrouped, _ = calculate_cost(get_cost, testSet, options, k=args.acc_at_k)
        buf = 'Evaluated model %s, k:%d, Test_Cost:%f, Test_Acc:%f, Test_Acc_Grouped:%s' % (best_model_name, args.acc_at_k, testCost, testAcc, str(testAccGrouped))
        print buf
        print2file(buf, logFile)
        pdb.set_trace()

    print 'Optimization start !!'
    bestTrainCost = 0.0
    bestValidCost = 100000.0
    #bestValidCost = -100000.0
    bestTestCost = 0.0
    bestValidAcc = 0.0
    bestTestAcc = 0.0
    bestValidAccGrouped = 0.0
    bestTestAccGrouped = 0.0
    epochDuration = 0.0
    bestEpoch = 0
    for epoch in xrange(max_epochs):
        iteration = 0
        costVec = []
        startTime = time.time()
        for index in random.sample(range(n_batches), n_batches):
            use_noise.set_value(1.)
            batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
            batchY = trainSet[1][index*batchSize:(index+1)*batchSize]
            x, y, mask, lengths = padMatrix(batchX, batchY, options)
            costValue = f_grad_shared(x, y, mask, lengths)
            f_update()
            costVec.append(costValue)

            if iteration % 100 == 0 and verbose:
                buf = 'Epoch:%d, Iteration:%d/%d, Train_Cost:%f' % (epoch, iteration, n_batches, costValue)
                print buf
            iteration += 1
        duration = time.time() - startTime
        use_noise.set_value(0.)
        trainCost = np.mean(costVec)
        validCost, validAcc, validAccGrouped, _ = calculate_cost(get_cost, validSet, options, k=args.acc_at_k)
        testCost, testAcc, testAccGrouped, emb = calculate_cost(get_cost, testSet, options, k=args.acc_at_k)
        buf = 'Epoch:%d, Duration:%f, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f, Valid_Acc:%f, Test_Acc:%f, Valid_Acc_Grouped:%s, Test_Acc_Grouped:%s' % (epoch, duration, trainCost, validCost, testCost, validAcc, testAcc, str(validAccGrouped), str(testAccGrouped))
        print buf
        print2file(buf, logFile)
        epochDuration += duration
        if validCost < bestValidCost:
        #if validAccGrouped[0] > bestValidCost:
            bestValidCost = validCost
            #bestValidCost = validAccGrouped[0]
            bestTestCost = testCost
            bestValidAcc = validAcc
            bestTestAcc = testAcc
            bestValidAccGrouped = validAccGrouped
            bestTestAccGrouped = testAccGrouped
            bestTrainCost = trainCost
            bestEpoch = epoch
            tempParams = unzip(tparams)
            np.savez_compressed(outFile + '.' + str(epoch), **tempParams)
            np.save(outFile + '.emb',  emb)
    buf = 'Best Epoch:%d, Avg_Duration:%f, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f, Valid_Acc:%f, Test_Acc:%f, Valid_Acc_Grouped:%s, Test_Acc_Grouped:%s' % (bestEpoch, epochDuration/max_epochs, bestTrainCost, bestValidCost, bestTestCost, bestValidAcc, bestTestAcc, str(bestValidAccGrouped), str(bestTestAccGrouped))
    print buf
    print2file(buf, logFile)

def parse_arguments(parser):
    parser.add_argument('data_dir', type=str, metavar='<data_dir>', default='data/gram_seqs/', help='The directory of data (seq_file, label_file, tree_file, embed_file)')
    parser.add_argument('seq_file', type=str, metavar='<visit_file>', help='The name of the Pickled file containing visit information of patients')
    parser.add_argument('label_file', type=str, metavar='<label_file>', help='The name of the Pickled file containing label information of patients')
    parser.add_argument('tree_file', type=str, metavar='<tree_file>', help='The name of the Pickled files containing the ancestor information of the input medical codes. Only use the prefix and exclude ".level#.pk".')
    parser.add_argument('out_dir', metavar='<out_dir>', default='result/trials/', help='The path to save the output models and results. The models will be saved after every epoch')
    parser.add_argument('--embed_file', type=str, default='', help='The name of the Pickled file containing the representation vectors of medical codes. If you are not using medical code representations, do not use this option')
    parser.add_argument('--embed_size', type=int, default=400, help='The dimension size of the visit embedding. If you are providing your own medical code vectors, this value will be automatically decided. (default value: 400)')
    parser.add_argument('--p2c_file', type=str, default='', help='The name of the Pickled file containing the parents to children (p2c) and children to parents (c2p) mappings')
    parser.add_argument('--fixed_embed', action='store_true', help='Disable embedding training, use fixed initial embedding constantly')
    parser.add_argument('--rnn_size', type=int, default=400, help='The dimension size of the hidden layer of the GRU (default value: 400)')
    parser.add_argument('--no_attention', action='store_true', help='Disable attention, directly sum up all ancestor embeddings')
    parser.add_argument('--leaf', action='store_true', help='Only use leaf embeddings without attention sum of ancestors')
    parser.add_argument('--rollup', action='store_true', help='All leaves use their upper-level node embeddings')
    parser.add_argument('--attention_size', type=int, default=100, help='The dimension size of hidden layer of the MLP that generates the attention weights (default value: 100)')
    parser.add_argument('--max_level', type=int, default=5, help='the maximum levels of hierarchy, 5 for ccs_dx, 4 for ccs_pr')
    parser.add_argument('--level', type=int, default=None, help='how many levels of hierarchy to use, if None, set to max_level')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs (default value: 50)')
    parser.add_argument('--L2', type=float, default=0.001, help='L2 regularization coefficient for all weights except RNN (default value: 0.001)')
    parser.add_argument('--dropout_rate', type=float, default=0.6, help='Dropout rate used for the hidden layer of RNN (default value: 0.6)')
    parser.add_argument('--log_eps', type=float, default=1e-8, help='A small value to prevent log(0) (default value: 1e-8)')
    parser.add_argument('--verbose', action='store_true', help='Print output after every 100 mini-batches (default false)')
    parser.add_argument('--seed', type=int, default=0, help='The random seed to split datasets')
    parser.add_argument('--acc_at_k', type=int, default=20, help='Accuracy@k')
    parser.add_argument('--test', action='store_true', help='if True, load the best model in current directory and test')
    args = parser.parse_args()
    return args

def calculate_dimSize(seqFile):
    seqs = pickle.load(open(seqFile, 'rb'))
    codeSet = set()
    for patient in seqs:
        for visit in patient:
            for code in visit:
                codeSet.add(code)
    return max(codeSet) + 1

def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    rootCode = tree.values()[0][1]
    return rootCode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    args.seq_file = args.data_dir + args.seq_file
    args.label_file = args.data_dir + args.label_file
    args.tree_file = args.data_dir + args.tree_file
    if args.embed_file:
        args.embed_file = args.data_dir + args.embed_file
    if args.p2c_file:
        args.p2c_file = args.data_dir + args.p2c_file

    args.out_dir = args.out_dir[:-1] + '_s' + str(args.seed) + '/'
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir) 
    args.out_file = args.out_dir + 'result'
    copy('gram.py', args.out_dir)
    cmd_input = 'python ' + ' '.join(sys.argv)
    with open(os.path.join(args.out_dir, 'cmd_input.txt'), 'w') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')
    if args.level is None:
        args.level = args.max_level

    inputDimSize = calculate_dimSize(args.seq_file)
    numClass = calculate_dimSize(args.label_file)
    numAncestors = get_rootCode(args.tree_file+'.level3.pk') - inputDimSize + 1

    train_GRAM(
        seqFile=args.seq_file, 
        inputDimSize=inputDimSize,
        treeFile=args.tree_file,
        numAncestors=numAncestors, 
        labelFile=args.label_file, 
        numClass=numClass,
        outFile=args.out_file, 
        embFile=args.embed_file, 
        embDimSize=args.embed_size, 
        p2cFile=args.p2c_file, 
        hiddenDimSize=args.rnn_size,
        attentionDimSize=args.attention_size,
        batchSize=args.batch_size, 
        max_epochs=args.n_epochs, 
        L2=args.L2, 
        dropoutRate=args.dropout_rate, 
        logEps=args.log_eps, 
        verbose=args.verbose
    )
