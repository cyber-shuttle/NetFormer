import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import r2_score
import pickle
from STDP_helper import *
from netformer_STDP import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--totalsteps', type=int, default=100000)  # total simulation timesteps
    parser.add_argument('--npre', type=int, default=100)  # number of presynaptic neurons
    parser.add_argument('--rate', type=int, default=50)  # Poisson rate for each presynaptic spike train
    parser.add_argument('--histlen', type=int, default=5)  # Poisson rate for each presynaptic spike train
    parser.add_argument('--standardize', type=int, default=1)  # if 1, standardize v using training mean & std
    parser.add_argument('--embdim', type=int, default=101)  # concat embedding dim
    parser.add_argument('--projdim', type=int, default=0)  # key/query proj dim. If 0, set to None
    parser.add_argument('--maxepoch', type=int, default=3)  # number of training epochs
    parser.add_argument('--batchsize', type=int, default=64)  # batch size
    parser.add_argument('--lr', type=float, default=0.005)  # learning rate
    parser.add_argument('--smoothlen', type=int, default=10000)  # learning rate
    parser.add_argument('--lrschr', type=int, default=0)  # if 1, use learning rate scheduler
    # parser.add_argument('--lrstep', type=int, default=1)
    # parser.add_argument('--lrgamma', type=float, default=0.99)
    parser.add_argument('--outdir', type=str, default='./')
    parser.add_argument('--seeds', type=int, nargs="+", default=[0, 1, 2, 3, 4])

    args = parser.parse_args()
    paramdict = vars(parser.parse_args())
    expname = ''
    for k, v in paramdict.items():
        if k != 'outdir':
            expname += k
            if k == 'seeds':
                expname += ','.join(map(str, v))
            else:
                expname += str(v)
            expname += '_'
    expname = expname.strip('_')
    print(expname)

    def align_with_hist(curr_inputs, nhist):
        # curr_inputs: (nsamples, nvar, 1). return: res: (nsamples-nhist, nvar, nhist+1)
        # res[:, :, -1] store the most recent timesteps
        res = [curr_inputs]
        for i in range(1, nhist + 1):
            inputs1 = np.roll(curr_inputs, -i, 0)
            res.append(inputs1)
        res = np.concatenate(res, axis=-1)
        return res[:curr_inputs.shape[0]-nhist]

    # def smooth(y, box_pts):
    #     box = np.ones(box_pts) / box_pts
    #     if len(y.shape) == 2:
    #         y_smooth = np.array([np.convolve(y[i], box, mode='valid') for i in range(len(y))])
    #     else:
    #         y_smooth = np.convolve(y, box, mode='valid')
    #     return y_smooth

    def smooth(y, box_pts):
        if len(y.shape) == 2:
            cumsum_y = np.cumsum(np.insert(y, 0, 0, axis=1), axis=1)
            return (cumsum_y[:, box_pts:] - cumsum_y[:, :-box_pts])/box_pts
        else:
            cumsum_y = np.cumsum(np.insert(y, 0, 0))
            return (cumsum_y[box_pts:] - cumsum_y[:-box_pts])/box_pts

    ###### simulate data #####
    pars = default_pars_STDP(T=args.totalsteps, dt=1.)  # Simulation duration 200 ms
    pars['gE_bar'] = 0.024  # max synaptic conductance
    pars['gE_init'] = 0.014  # initial synaptic conductance
    pars['VE'] = 0.  # [mV] Synapse reversal potential
    pars['tau_syn_E'] = 5.  # [ms] EPSP time constant

    # generate Poisson type spike trains
    pre_spike_train_ex = Poisson_generator(pars, rate=args.rate, n=args.npre, myseed=2020)  # pre_spike_train_ex: (npre, totalsteps)
    # simulate the LIF neuron and record the synaptic conductance
    v, rec_spikes, gE, P, M, gE_bar_update = run_LIF_cond_STDP(pars, pre_spike_train_ex)  # v: (totalsteps, )

    ##### generate training/test samples #####
    ntrain = int(args.totalsteps*0.8)
    if args.standardize:
        v_train_mean = np.mean(v[:ntrain])
        v_train_std = np.std(v[:ntrain])
        v = (v - v_train_mean)/v_train_std
    v_train = v[:ntrain]
    v_test = v[ntrain-args.histlen:]

    pre_spk_train = pre_spike_train_ex[:, :ntrain]
    pre_spk_test = pre_spike_train_ex[:, ntrain-args.histlen:]

    x_train = np.vstack((v_train, pre_spk_train)).T  # (ntrain, 1+npre)
    x_train = np.expand_dims(x_train, axis=-1)  # (ntrain, 1+npre, 1)
    x_train_align = align_with_hist(x_train, args.histlen)
    x_train_tag = x_train_align[:, 0, -1]  # training targets
    x_train_inp = x_train_align[:, :, :-1]  # training inputs
    assert np.sum(np.abs(x_train_tag - v_train[args.histlen:])) == 0
    del x_train_align

    x_test = np.vstack((v_test, pre_spk_test)).T
    x_test = np.expand_dims(x_test, axis=-1)
    x_test_align = align_with_hist(x_test, args.histlen)
    x_test_tag = x_test_align[:, 0, -1]  # test targets
    x_test_inp = x_test_align[:, :, :-1]  # test inputs
    assert np.sum(np.abs(x_test_tag - v[ntrain:])) == 0
    del x_test_align

    print(x_train_inp.shape, x_train_tag.shape, x_test_inp.shape, x_test_tag.shape)

    x_train_inp = torch.from_numpy(x_train_inp).float()
    x_train_tag = torch.from_numpy(x_train_tag).float()
    x_test_inp = torch.from_numpy(x_test_inp).float()
    x_test_tag = torch.from_numpy(x_test_tag).float()
    all_inp = torch.cat((x_train_inp, x_test_inp), 0)

    train_dataset = TensorDataset(x_train_inp, x_train_tag)
    test_dataset = TensorDataset(x_test_inp, x_test_tag)
    print(len(train_dataset), len(test_dataset))

    gE_bar_smooth = smooth(gE_bar_update[:, args.histlen:], args.smoothlen)
    np.save(args.outdir + 'test_target', x_test_tag)
    np.save(args.outdir + 'gE_bar', gE_bar_update[:, args.histlen:])
    np.save(args.outdir + 'gE_bar_smooth', gE_bar_smooth)

    ##### train model and keep records #####
    train_mse_allseeds = {}
    test_mse_allseeds = {}
    test_r2_allseeds = {}
    ccs_allseeds = {}

    for seed in args.seeds:
        print(seed)
        train_mse_allepochs = []

        torch.manual_seed(seed)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        projdim = None
        if args.projdim > 0:
            projdim = args.projdim
        model_cat = TransformerForecasting_cat(seq_len=101, input_dim=args.histlen, emb_dim=args.embdim, proj_dim=projdim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_cat.parameters(), lr=args.lr)

        with torch.no_grad():
            test_pred = model_cat(x_test_inp)
            test_mse0 = criterion(test_pred, x_test_tag).item()

            train_pred = model_cat(x_train_inp)
            train_mse0 = criterion(train_pred, x_train_tag).item()
            train_mse_allepochs.append(train_mse0)
        print(f'epoch 0: train mse {train_mse0:.3f}    test mse {test_mse0:.3f}')

        # if args.lrschr:
        #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lrstep, gamma=args.lrgamma)
        # Training loop
        for epoch in range(1, args.maxepoch+1):
            train_loss_epoch = train(train_dataloader, model_cat, criterion, optimizer, max_grad_norm=None)
            train_mse_allepochs.append(train_loss_epoch)
            print(f'epoch {epoch}: train mse {train_mse_allepochs[-1]:.3f}')

            if epoch == args.maxepoch:
                with torch.no_grad():
                    test_pred, test_attn = model_cat.attention(x_test_inp)
                    test_pred = torch.squeeze(test_pred)
                    test_mse = criterion(test_pred, x_test_tag).item()

                    test_r2 = r2_score(x_test_tag.numpy(), test_pred.numpy())
                    _, all_attn = model_cat.attention(all_inp)
                    all_attn = torch.squeeze(all_attn)[:, 1:].numpy().T
                    all_attn_smooth = smooth(all_attn, args.smoothlen)
                    ccs = [np.corrcoef(all_attn_smooth[i], gE_bar_smooth[i])[0, 1] for i in range(100)]

                    print(f'test mse {test_mse:.3f} r2 {test_r2:.3f}   corr mean {np.mean(ccs):.3f}  median {np.median(ccs):.3f}')
                    ### saving
                    np.save(args.outdir + f'attn_allsteps_seed{seed}', all_attn)
                    np.save(args.outdir + f'test_pred_seed{seed}', test_pred)
                    np.save(args.outdir + f'test_attn_seed{seed}', test_attn)
                    torch.save(model_cat.state_dict(), args.outdir + f'trained_model_seed{seed}.pth')

            # if args.lrschr:
            #     scheduler.step()

        train_mse_allseeds[seed] = train_mse_allepochs
        test_mse_allseeds[seed] = test_mse
        test_r2_allseeds[seed] = test_r2
        ccs_allseeds[seed] = ccs

    res = {'train_mse_allseeds': train_mse_allseeds,
           'test_mse_allseeds': test_mse_allseeds,
           'test_r2_allseeds': test_r2_allseeds,
           'ccs_allseeds': ccs_allseeds
           }
    with open(args.outdir + 'final_scores.pkl', 'wb') as f:
        pickle.dump(res, f)


















