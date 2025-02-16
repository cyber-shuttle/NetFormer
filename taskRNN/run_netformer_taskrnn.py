import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import r2_score
import pickle
from scipy import stats
import os
from netformer_taskrnn import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnndim', type=int, default=12)  # RNN dimension
    parser.add_argument('--useinp', type=int, default=1)  # if 1, include stim inputs
    parser.add_argument('--histlen', type=int, default=5)  # number of history steps
    parser.add_argument('--LN', type=int, default=1)  # if 1, use layernorm
    parser.add_argument('--standardize', type=int, default=0)  # if 1, standardize signals using training mean & std
    parser.add_argument('--embdim', type=int, default=0)  # concat embedding dim. If 0, set to nvar
    parser.add_argument('--projdim', type=int, default=0)  # key/query proj dim. If 0, set to None
    parser.add_argument('--ptrain', type=float, default=0.8)  # fraction of trials for training
    parser.add_argument('--padstart', type=int, default=0)  # if 1, zero-pad invalid history
    parser.add_argument('--maxepoch', type=int, default=10)  # number of training epochs
    parser.add_argument('--batchsize', type=int, default=64)  # batch size
    parser.add_argument('--lr', type=float, default=0.0025)  # learning rate
    parser.add_argument('--lrschr', type=int, default=0)  # if 1, use learning rate scheduler
    parser.add_argument('--lrstep', type=int, default=2)
    parser.add_argument('--lrgamma', type=float, default=0.9)
    parser.add_argument('--datapath', type=str, default='../taskRNN_data/DelayComparison/')  # tasks: DelayComparison, GoNogo, PerceptualDecisionMaking
    parser.add_argument('--outdir', type=str, default='../taskRNN_results/DelayComparison_results/')
    parser.add_argument('--seeds', type=int, nargs="+", default=[0, 1, 2, 3, 4])

    args = parser.parse_args()
    paramdict = vars(parser.parse_args())
    expname = ''
    for k, v in paramdict.items():
        if k not in ['datapath', 'outdir']:
            expname += k
            if k == 'seeds':
                expname += ','.join(map(str, v))
            else:
                expname += str(v)
            expname += '_'
    expname = expname.strip('_')
    print(expname)
    if args.save:
        os.mkdir(f'{args.outdir}{expname}')

    def make_data(rnn_activity, rnn_inputs=None, histlen=1, padstart=False, standardize=False):
        ntrial = rnn_activity.shape[0]
        triallen = rnn_activity.shape[1]
        nneur = rnn_activity.shape[2]

        if rnn_inputs is not None:
            rnn_inputs_ = np.zeros(rnn_inputs.shape)
            rnn_inputs_[:, :-1, :] = rnn_inputs[:, 1:, :]
            rnn_activity_inp = np.concatenate((rnn_activity, rnn_inputs_), axis=-1)
        else:
            rnn_activity_inp = rnn_activity
        if standardize:
            # Compute the mean and standard deviation along the first dimension
            mean = np.mean(rnn_activity_inp, axis=1, keepdims=True)
            std = np.std(rnn_activity_inp, axis=1, keepdims=True)
            # Standardize the array
            rnn_activity_inp = (rnn_activity_inp - mean) / std

        nneurin = rnn_activity_inp.shape[2]

        if not padstart:  # start from the timestep with valid full history
            activity_aligned_alltrials = np.zeros((ntrial, triallen - histlen, nneurin, histlen))
            target_alltrials = np.zeros((ntrial, triallen - histlen, nneur, 1))
            for i in range(ntrial):
                activity_trial_i = rnn_activity_inp[i].T  # nneur x triallen
                for k in range(triallen - histlen):
                    activity_aligned_alltrials[i][k] = activity_trial_i[:, k:k + histlen]
                    target_alltrials[i][k, :, 0] = activity_trial_i[:nneur, k + histlen]

        else:  # start from the first timestep, pad invalid history with zero
            activity_aligned_alltrials = np.zeros((ntrial, triallen - 1, nneurin, histlen))
            target_alltrials = np.zeros((ntrial, triallen - 1, nneur, 1))
            for i in range(ntrial):
                padding = np.zeros((nneurin, histlen - 1))
                activity_trial_i = rnn_activity_inp[i].T  # nneur x triallen
                pad_activity_trial_i = np.hstack((padding, activity_trial_i))
                for k in range(triallen - 1):
                    activity_aligned_alltrials[i][k] = pad_activity_trial_i[:, k + histlen - histlen:k + histlen]
                    target_alltrials[i][k, :, 0] = activity_trial_i[:nneur, k + 1]

        return activity_aligned_alltrials, target_alltrials  # activity: (ntrial, samples per trial, nneurin, histlen), target: (ntrial, samples per trial, nneurin, 1)


    ###### load data #####
    rnn_activity = np.load(f'{args.datapath}rnn{args.rnndim}/hidden_activity_alltrials.npy')  # num of trials x trial length x num of neurons
    GT = np.load(f'{args.datapath}rnn{args.rnndim}/W_hidden_GT.npy')
    total_trials = rnn_activity.shape[0]
    n_train_trials = int(total_trials*args.ptrain)
    train_trials = rnn_activity[:n_train_trials]
    test_trials = rnn_activity[n_train_trials:]

    if args.useinp:
        rnn_inputs = np.load(f'{args.datapath}rnn{args.rnndim}/inputs_alltrials.npy')  # num of trials x trial length x stim dim
        train_stim = rnn_inputs[:n_train_trials]
        test_stim = rnn_inputs[n_train_trials:]
        print(train_trials.shape, test_trials.shape, train_stim.shape, test_stim.shape)
    else:
        train_stim = None
        test_stim = None
        print(train_trials.shape, test_trials.shape)

    ##### generate training/test samples #####
    train_activity_aligned_alltrials, train_target_alltrials = make_data(train_trials, rnn_inputs=train_stim, histlen=args.histlen,
                                                                       padstart=args.padstart)
    print(train_activity_aligned_alltrials.shape, train_target_alltrials.shape)
    inputs_train = train_activity_aligned_alltrials.reshape((train_activity_aligned_alltrials.shape[0] *
                                                           train_activity_aligned_alltrials.shape[1],
                                                           train_activity_aligned_alltrials.shape[2],
                                                           train_activity_aligned_alltrials.shape[3]))
    targets_train = train_target_alltrials.reshape((train_target_alltrials.shape[0] * train_target_alltrials.shape[1],
                                                  train_target_alltrials.shape[2], train_target_alltrials.shape[3]))

    print(inputs_train.shape, targets_train.shape)

    inputs_train = torch.from_numpy(inputs_train).float()
    targets_train = torch.from_numpy(targets_train).float()
    train_dataset = TensorDataset(inputs_train, targets_train)
    print(len(train_dataset))

    test_activity_aligned_alltrials, test_target_alltrials = make_data(test_trials, rnn_inputs=test_stim, histlen=args.histlen,
                                                                       padstart=args.padstart)
    print(test_activity_aligned_alltrials.shape, test_target_alltrials.shape)
    inputs_test = test_activity_aligned_alltrials.reshape((test_activity_aligned_alltrials.shape[0] *
                                                           test_activity_aligned_alltrials.shape[1],
                                                           test_activity_aligned_alltrials.shape[2],
                                                           test_activity_aligned_alltrials.shape[3]))
    targets_test = test_target_alltrials.reshape((test_target_alltrials.shape[0] * test_target_alltrials.shape[1],
                                                  test_target_alltrials.shape[2], test_target_alltrials.shape[3]))

    print(inputs_test.shape, targets_test.shape)

    inputs_test = torch.from_numpy(inputs_test).float()
    targets_test = torch.from_numpy(targets_test).float()
    test_dataset = TensorDataset(inputs_test, targets_test)
    print(len(test_dataset))

    ##### train model and keep records #####
    train_mse_allseeds = {}
    train_r2_allseeds = {}
    test_mse_allseeds = {}
    test_r2_allseeds = {}
    spear_cc_allseeds = {}
    pearson_cc_allseeds = {}
    offdiag_mask = ~np.eye(args.rnndim, dtype=bool)

    for seed in args.seeds:
        print(seed)
        train_mse_allepochs = []

        torch.manual_seed(seed)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        projdim = None
        embdim = inputs_train.shape[1]
        nq = args.rnndim
        if args.projdim > 0:
            projdim = args.projdim
        if args.embdim > 0:
            embdim = args.embdim
        print(embdim, projdim, nq)

        model_cat = TransformerForecasting_cat(seq_len=inputs_train.shape[1], input_dim=args.histlen, emb_dim=embdim, nq=nq, proj_dim=projdim, use_LN=args.LN)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_cat.parameters(), lr=args.lr)

        with torch.no_grad():
            test_pred = model_cat(inputs_test)
            test_mse0 = criterion(test_pred, targets_test).item()

            train_pred = model_cat(inputs_train)
            train_mse0 = criterion(train_pred, targets_train).item()
            train_mse_allepochs.append(train_mse0)
        print(f'epoch 0: train mse {train_mse0:.3f}    test mse {test_mse0:.3f}')

        if args.lrschr:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lrstep, gamma=args.lrgamma)
        # Training loop
        for epoch in range(1, args.maxepoch+1):
            train_loss_epoch = train(train_dataloader, model_cat, criterion, optimizer, max_grad_norm=None)
            train_mse_allepochs.append(train_loss_epoch)

            if epoch == args.maxepoch:
                test_pred, test_attn = model_cat.attention(inputs_test)
                test_mse = criterion(test_pred, targets_test).item()
                test_r2 = r2_score(targets_test.numpy()[:,:,0], test_pred.numpy()[:,:,0])
                avg_attn = np.mean(test_attn.numpy(), axis=0)[:, :nq]
                spear_cc = stats.spearmanr(GT[offdiag_mask].flatten(), avg_attn[offdiag_mask].flatten()).correlation
                pearson_cc = np.corrcoef(GT[offdiag_mask].flatten(), avg_attn[offdiag_mask].flatten())[0, 1]
                print(f'epoch {epoch}: test mse {test_mse:.3f}   r2 {test_r2:.3f}   spearmancc {spear_cc:.3f}    pearsoncc {pearson_cc:.3f}')
                ### saving
                np.save(f'{args.outdir}{expname}/test_attn_seed{seed}', test_attn.numpy())
                np.save(f'{args.outdir}{expname}/test_pred_seed{seed}', test_pred)
                torch.save(model_cat.state_dict(), f'{args.outdir}{expname}/trained_model_seed{seed}.pth')

            if args.lrschr:
                scheduler.step()
                # print(scheduler.get_last_lr())


        train_mse_allseeds[seed] = train_mse_allepochs
        test_mse_allseeds[seed] = test_mse
        test_r2_allseeds[seed] = test_r2
        spear_cc_allseeds[seed] = spear_cc
        pearson_cc_allseeds[seed] = pearson_cc

    res = {'train_mse_allseeds': train_mse_allseeds,
           'test_mse_allseeds': test_mse_allseeds,
           'test_r2_allseeds': test_r2_allseeds,
           'spear_cc_allseeds': spear_cc_allseeds,
           'pearson_cc_allseeds': pearson_cc_allseeds,
           }
    with open(f'{args.outdir}{expname}/res.pkl', 'wb') as f:
        pickle.dump(res, f)