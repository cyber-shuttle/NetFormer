import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import r2_score
import pickle
from scipy import stats
import os
from netformer_fiete import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--totalsteps', type=int, default=100000)  # total simulation timesteps
    parser.add_argument('--skipsteps', type=int, default=10000)  # number of initial timesteps to skip
    parser.add_argument('--r', type=str, default='0.025')  # recurrent strength
    parser.add_argument('--useinp', type=int, default=0)  # if 1, include external input
    parser.add_argument('--histlen', type=int, default=1)  # number of history steps
    parser.add_argument('--LN', type=int, default=0)  # if 1, use layernorm
    parser.add_argument('--standardize', type=int, default=0)  # if 1, standardize signals using training mean & std
    parser.add_argument('--embdim', type=int, default=100)  # concat embedding dim. If 0, set to nvar
    parser.add_argument('--projdim', type=int, default=0)  # key/query proj dim. If 0, set to None
    parser.add_argument('--ptrain', type=float, default=0.8)  # fraction of trials for training
    parser.add_argument('--maxepoch', type=int, default=100)  # number of training epochs
    parser.add_argument('--batchsize', type=int, default=32)  # batch size
    parser.add_argument('--lr', type=float, default=0.001)  # learning rate
    parser.add_argument('--lrschr', type=int, default=0)  # if 1, use learning rate scheduler
    parser.add_argument('--lrstep', type=int, default=1)
    parser.add_argument('--lrgamma', type=float, default=1)
    parser.add_argument('--usegpu', type=int, default=1)
    parser.add_argument('--datapath', type=str, default='../Fiete_data/')  # tasks: DelayComparison, GoNogo, PerceptualDecisionMaking
    parser.add_argument('--outdir', type=str, default='../Fiete_results_poisson/')
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs="+", default=[0, 1, 2])

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

    def align_with_hist(curr_inputs, nhist):
        # curr_inputs: (nsamples, nvar, 1). return: res: (nsamples-nhist, nvar, nhist+1)
        # res[:, :, -1] store the most recent timesteps
        res = [curr_inputs]
        for i in range(1, nhist + 1):
            inputs1 = np.roll(curr_inputs, -i, 0)
            res.append(inputs1)
        res = np.concatenate(res, axis=-1)
        return res[:curr_inputs.shape[0] - nhist]

    ###### load data #####
    activity = np.load(f'{args.datapath}activation_r{args.r}.npy')  # 1,000,000 x 100  (total steps x neurons). One step ahead of spikes. activity[k] corresponds to spikes[k+1]
    spikes = np.load(f'{args.datapath}spikes_r{args.r}.npy')  # 1,000,000 x 100  (total steps x neurons).
    nneur = activity.shape[1]
    GT = np.load(f'{args.datapath}W_r{args.r}.npy')
    totalsteps = min(args.totalsteps, activity.shape[0]-args.skipsteps)
    activity = activity[args.skipsteps:args.skipsteps+totalsteps, :]
    spikes = spikes[args.skipsteps:args.skipsteps+totalsteps, :]

    # use gpu if possible
    device = "cpu"
    if args.usegpu:
        if torch.cuda.is_available():
            device = 'cuda:0'
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
    print('device:', device)
    ##### generate training/test samples #####
    ntrain = int(totalsteps * args.ptrain)
    if args.standardize:
        act_train_mean = np.mean(activity[:ntrain], axis=0)
        act_train_std = np.std(activity[:ntrain], axis=0)
        act_train_std[act_train_std == 0.0] = 1.0
        activity -= act_train_mean
        activity /= act_train_std
        # assert np.isfinite(activity).all()
    # plt.plot(activity[:, 0])
    # plt.show()

    if args.useinp:
        inp = np.full((activity.shape[0], 1), 0.001)
        activity = np.hstack((activity, inp))

    activity_train = activity[:ntrain]
    activity_test = activity[ntrain-args.histlen:]
    spikes_train = spikes[:ntrain]
    spikes_test = spikes[ntrain-args.histlen:]

    x_train = np.expand_dims(activity_train, axis=-1)  # (ntrain, nneur+1, 1)
    x_train_align = align_with_hist(x_train, args.histlen)  #(ntrain-nhist, nneur+1, nhist+1)
    x_train_inp = x_train_align[:, :, :-1]  # training inputs (ntrain-nhist, nneur+1, nhist)
    # x_train_tag = x_train_align[:, :nneur, -1]  # training targets (ntrain-nhist, nneur)
    # assert np.sum(np.abs(x_train_tag - activity_train[args.histlen:, :nneur])) == 0
    del x_train_align
    x_train_tag = spikes_train[args.histlen:, :nneur]
    x_train_tag = np.expand_dims(x_train_tag, axis=-1)  # (ntrain-nhist, nneur, 1)

    x_test = np.expand_dims(activity_test, axis=-1)
    x_test_align = align_with_hist(x_test, args.histlen)
    x_test_inp = x_test_align[:, :, :-1]  # test inputs
    # x_test_tag = x_test_align[:, :nneur, -1]  # test targets
    # assert np.sum(np.abs(x_test_tag - activity[ntrain:, :nneur])) == 0
    del x_test_align
    x_test_tag = spikes[ntrain:, :nneur]
    x_test_tag = np.expand_dims(x_test_tag, axis=-1)

    print(x_train_inp.shape, x_train_tag.shape, x_test_inp.shape, x_test_tag.shape)

    x_train_inp = torch.from_numpy(x_train_inp).float()
    x_train_tag = torch.from_numpy(x_train_tag).float()
    x_test_inp = torch.from_numpy(x_test_inp).float()
    x_test_tag = torch.from_numpy(x_test_tag).float()

    train_dataset = TensorDataset(x_train_inp, x_train_tag)
    test_dataset = TensorDataset(x_test_inp, x_test_tag)
    print(len(train_dataset), len(test_dataset))

    ##### train model and keep records #####
    train_loss_allepochs_allseeds = {}
    test_loss_allepochs_allseeds = {}
    offdiag_mask = ~np.eye(nneur, dtype=bool)

    for seed in args.seeds:
        print(seed)
        torch.manual_seed(seed)

        train_loss_allepochs = []
        test_loss_allepochs = []

        train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)

        projdim = None
        embdim = x_train_inp.shape[1]
        if args.projdim > 0:
            projdim = args.projdim
        if args.embdim > 0:
            embdim = args.embdim
        print(embdim, projdim)

        model_cat = TransformerForecasting_cat(seq_len=x_train_inp.shape[1], input_dim=args.histlen, emb_dim=embdim, nq=nneur, proj_dim=projdim, use_LN=args.LN).to(device)
        attn_shape = (nneur, x_train_inp.shape[1])
        # criterion = nn.MSELoss()
        criterion = nn.PoissonNLLLoss()
        optimizer = optim.Adam(model_cat.parameters(), lr=args.lr)

        test_loss_allepochs.append(evaluate(test_dataloader, model_cat, criterion, device=device, return_avg_attn=False))
        train_loss_allepochs.append(evaluate(train_dataloader, model_cat, criterion, device=device, return_avg_attn=False))
        print(f'epoch 0: train loss {train_loss_allepochs[-1]:.3f}    test loss {test_loss_allepochs[-1]:.3f}')

        if args.lrschr:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lrstep, gamma=args.lrgamma)
        # Training loop
        for epoch in range(1, args.maxepoch+1):
            train_loss_epoch = train(train_dataloader, model_cat, criterion, optimizer, device=device, max_grad_norm=None)
            train_loss_allepochs.append(train_loss_epoch)

            test_loss_epoch, avg_attn = evaluate(test_dataloader, model_cat, criterion, device=device, return_avg_attn=True, attn_shape=attn_shape)
            test_loss_allepochs.append(test_loss_epoch)
            # avg_attn = avg_attn.cpu()[:, :nneur]

            print(f'epoch {epoch}: train loss {train_loss_allepochs[-1]:.3f}   test loss {test_loss_allepochs[-1]:.3f}')

            if args.lrschr:
                scheduler.step()
                # print(scheduler.get_last_lr())

        if args.save:
            avg_attn = avg_attn.cpu()[:, :nneur]
            np.save(f'{args.outdir}{expname}/avg_test_attn_seed{seed}', avg_attn.numpy())
            torch.save(model_cat.state_dict(), f'{args.outdir}{expname}/trained_model_seed{seed}.pth')

        train_loss_allepochs_allseeds[seed] = train_loss_allepochs
        test_loss_allepochs_allseeds[seed] = test_loss_allepochs

    res = {'train_loss_allepochs_allseeds': train_loss_allepochs_allseeds,
           'test_loss_allepochs_allseeds': test_loss_allepochs_allseeds,
           }
    with open(f'{args.outdir}{expname}.pkl', 'wb') as f:
        pickle.dump(res, f)


















