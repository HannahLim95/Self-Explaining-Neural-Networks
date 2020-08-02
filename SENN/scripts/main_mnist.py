# -*- coding: utf-8 -*-
""" Code for training and evaluating Self-Explaining Neural Networks.
Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
# Ignore Future Warnings (Joosje)
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

# Standard Imports
import sys, os
import numpy as np
import pdb
import pickle
import argparse
import operator
import matplotlib
import matplotlib.pyplot as plt

# Torch-related
import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader
from tqdm import tqdm


# Local imports
from SENN.arglist import get_senn_parser #parse_args as parse_senn_args
from SENN.models import GSENN
from SENN.conceptizers import image_fcc_conceptizer, image_cnn_conceptizer, input_conceptizer

from SENN.parametrizers import image_parametrizer
from SENN.aggregators import linear_scalar_aggregator, additive_scalar_aggregator
from SENN.trainers import HLearningClassTrainer, VanillaClassTrainer, GradPenaltyTrainer
from SENN.utils import plot_theta_stability, generate_dir_names, noise_stability_plots, concept_grid, plot_prob_drop
from SENN.eval_utils import estimate_dataset_lipschitz

from robust_interpret.explainers import gsenn_wrapper
from robust_interpret.utils import lipschitz_boxplot, lipschitz_argmax_plot, lipschitz_feature_argmax_plot


def revert_to_raw(t):
    return ((t*.3081) + .1307)

def load_mnist_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size = 64,
                    num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    train = MNIST('data/MNIST', train=True, download=True, transform=transform)
    test  = MNIST('data/MNIST', train=False, download=True, transform=transform)

    num_train = len(train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    # Create DataLoader
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler, **dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, valid_loader, test_loader, train, test

class new_wrapper(gsenn_wrapper):

    def compute_dataset_consistency(self,  dataset, reference_value = 0, inputs_are_concepts = True, save_path = None):
        """
            does compute_prob_drop for all dataset, returns stats and plots
        """
        drops = []
        atts  = []
        corrs = []
        i = 0
        print("shape of x:", dataset.shape)
        for x in dataset:
            if save_path:
                path = save_path + '_' + str(i)
            else:
                path = save_path
            i += 1
            p_d, att = self.compute_prob_drop(x, inputs_are_concepts = inputs_are_concepts, save_path = path)
            p_d = p_d.squeeze()
            att = att.squeeze()
            drops.append(p_d)
            atts.append(att)
            #pdb.set_trace()
            #assert len(p_d).shape[0] == atts.shape[0], "Attributions has wrong size"
            #pdb.set_trace()

            # print("attributions: ", atts)
            corrs.append(np.corrcoef(p_d, att)[0,1]) # Compute correlation per sample, then aggreate

        corrs = np.array(corrs)
        # pdb.set_trace()
        # drops = np.stack(drops)
        # atts  = np.stack(atts)
        #
        # np.corrcoef(drops.flatten(), atts.flatten())
        return corrs

    def compute_prob_drop(self, x, reference_value = 0, plot = False, save_path = None, inputs_are_concepts = True):
        """
            This is placed here to prevent having to update the robust_interpret package. 
        """
        print("x size in compute prob drop: ", x.size())
        # First, turn inputs into concepts
        if not inputs_are_concepts:
            # x = x.type(torch.FloatTensor)
            x = x.unsqueeze(dim = 0)
            h_x = self.net.forward(x, h_options = -1)
            # print("h_x: ",h_x)
            f = self.net.forward(x, h_x = h_x, h_options = 1)
        # Then, use concepts to forward pass through the model
        # if inputs_are_concepts:
        else:
            f   = self.net.forward(x.reshape(1,-1)) # model is compute_proba function, not neural model - we need to add output layer to compute probabilities. Not necessary for now TODO
        # So for now, we use self.net
        # else:
        #     f = self.model.forward(x.reshape(1,-1), h_x = h_x,  h_options = 1)
        pred_class = f.argmax()
        attributions = self(x) # attributions are theta values (i think)
        deltas = []
        for i in tqdm(range(h_x.shape[1])):
            x_p = x.clone()
            # x_p[i] = reference_value # uncomment this to be compatible with uci dataset
            h_x_p = h_x.clone()
            h_x_p[:,i, :] = reference_value
            if inputs_are_concepts:
                f_p = self.net(x_p.reshape(1,-1))
            else:
                f_p = self.net.forward(x, h_x = h_x_p,  h_options = 1)
            # print("outcome: ", f_p, f)
            delta_i = (f - f_p)[0,pred_class]
            # print("delta_i: ", delta_i)
            deltas.append(delta_i.cpu().detach().numpy())
        prob_drops = np.array(deltas)
        plot = True
        if plot:
            plot_prob_drop(attributions[0], prob_drops, save_path = save_path) # remove [0] after attributions for uci
        return prob_drops, attributions

def parse_args():
    senn_parser = get_senn_parser()

    ### Local ones
    parser = argparse.ArgumentParser(parents =[senn_parser],add_help=False,
        description='Interpteratbility robustness evaluation on MNIST')

    # #setup
    parser.add_argument('-d','--datasets', nargs='+',
                        default = ['heart', 'ionosphere', 'breast-cancer','wine','heart',
                        'glass','diabetes','yeast','leukemia','abalone'], help='<Required> Set flag')
    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=0.01,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_points', type=int, default=100,
                        help='sample size for dataset Lipschitz estimation')
    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')

    #####

    args = parser.parse_args()

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


def main():
    args = parse_args()
    args.nclasses = 10
    args.theta_dim = args.nclasses

    model_path, log_path, results_path = generate_dir_names('mnist', args)

    train_loader, valid_loader, test_loader, train_tds, test_tds = load_mnist_data(
                        batch_size=args.batch_size,num_workers=args.num_workers
                        )

    if args.h_type == 'input':
        conceptizer  = input_conceptizer()
        args.nconcepts = 28*28 + int(not args.nobias)
    elif args.h_type == 'cnn':
        
        
        #args.nconcepts +=     int(not args.nobias)
        conceptizer  = image_cnn_conceptizer(28*28, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)
    else:
        #args.nconcepts +=     int(not args.nobias)
        conceptizer  = image_fcc_conceptizer(28*28, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)


    parametrizer = image_parametrizer(28*28, args.nconcepts, args.theta_dim,  only_positive = args.positive_theta)

    aggregator   = additive_scalar_aggregator(args.concept_dim, args.nclasses)

    model        = GSENN(conceptizer, parametrizer, aggregator) #, learn_h = args.train_h)


    if args.load_model:
        checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']

    if args.theta_reg_type in ['unreg','none', None]:
        trainer = VanillaClassTrainer(model, args)
    elif args.theta_reg_type == 'grad1':
        trainer = GradPenaltyTrainer(model, args, typ = 1)
    elif args.theta_reg_type == 'grad2':
        trainer = GradPenaltyTrainer(model, args, typ = 2)
    elif args.theta_reg_type == 'grad3':
        trainer = GradPenaltyTrainer(model, args, typ = 3)
    elif args.theta_reg_type == 'crosslip':
        trainer = CLPenaltyTrainer(model, args)
    else:
        raise ValueError('Unrecoginzed theta_reg_type')

    if not args.load_model and args.train:
        trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)
        trainer.plot_losses(save_path=results_path)
    else:
        checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']
        trainer =  VanillaClassTrainer(model, args)

    trainer.validate(test_loader, fold = 'test')

    All_Results = {}

    


    ### 1. Single point lipshiz estimate via black box optim
    # All methods tested with BB optim for fair comparison)
    features = None
    classes = [str(i) for i in range(10)]
    model.eval()
    expl = new_wrapper(model,
                        mode      = 'classification',
                        input_type = 'image',
                        multiclass=True,
                        feature_names = features,
                        class_names   = classes,
                        train_data      = train_loader,
                        skip_bias = True,
                        verbose = False)



    #### Debug single input
    # x = next(iter(train_tds))[0]
    # attr = expl(x, show_plot = False)
    # pdb.set_trace()

    # #### Debug multi input
    # x = next(iter(test_loader))[0] # Transformed
    # x_raw = test_loader.dataset.test_data[:args.batch_size,:,:]
    # attr = expl(x, x_raw = x_raw, show_plot = True)
    # #pdb.set_trace()
    

    ### Consistency analysis

    # for i, (inputs, targets) in enumerate(test_loader):
    #         # get the inputs
    #         if model.cuda:
    #             inputs, targets = inputs.cuda(), targets.cuda()
    #         input_var = torch.autograd.Variable(inputs, volatile=True)
            
    #         save_path = results_path + '/faithfulness' + str(i) + '/'
    #         if not os.path.isdir(save_path):
    #             os.mkdir(save_path)
    #         corrs = expl.compute_dataset_consistency(input_var, inputs_are_concepts = False, save_path = save_path)
            


    # #### Debug argmax plot_theta_stability
    if args.h_type == 'input':
        x = next(iter(test_tds))[0].numpy()
        y = next(iter(test_tds))[0].numpy()
        x_raw = (test_tds.test_data[0].float()/255).numpy()
        y_raw = revert_to_raw(x)
        att_x = expl(x, show_plot = False)
        att_y = expl(y, show_plot = False)
        lip = 1
        lipschitz_argmax_plot(x_raw, y_raw, att_x,att_y, lip)# save_path=fpath)

        #pdb.set_trace()


    ### 2. Single example lipschitz estimate with Black Box
    do_bb_stability_example = False # Aangepast: was False
    if do_bb_stability_example:
        print('**** Performing lipschitz estimation for a single point ****')

        idx = 0
        print('Example index: {}'.format(idx))
        #x = train_tds[idx][0].view(1,28,28).numpy()
        x = next(iter(test_tds))[0].numpy()
        x_raw = (test_tds.test_data[0].float()/255).numpy()
        #x_raw = next(iter(train_tds))[0]

        # args.optim     = 'gp'
        # args.lip_eps   = 0.1
        # args.lip_calls = 10
        Results = {}

        lip, argmax = expl.local_lipschitz_estimate(x, bound_type='box_std',
                                                optim=args.optim,
                                                eps=args.lip_eps,
                                                n_calls=4*args.lip_calls,
                                                njobs = 1,
                                                verbose=2)
        #pdb.set_trace()
        Results['lip_argmax'] = (x, argmax, lip)
        # .reshape(inputs.shape[0], inputs.shape[1], -1)
        att = expl(x, None, show_plot=False)#.squeeze()
        # .reshape(inputs.shape[0], inputs.shape[1], -1)
        att_argmax = expl(argmax, None, show_plot=False)#.squeeze()

        #pdb.set_trace()
        Argmax_dict = {'lip': lip, 'argmax': argmax, 'x': x}
        fpath = os.path.join(results_path, 'argmax_lip_gp_senn.pdf')
        if args.h_type == 'input':
            lipschitz_argmax_plot(x_raw, revert_to_raw(argmax), att, att_argmax, lip, save_path=fpath)
        pickle.dump(Argmax_dict, open(
            results_path + '/argmax_lip_gp_senn.pkl', "wb"))
        pdb.set_trace()
        # print(asd.asd)

    noise_stability_plots(model, test_tds, cuda = args.cuda, save_path = results_path)
    
    ### 3. Local lipschitz estimate over multiple samples with Black BOx Optim
    do_bb_stability = False # Aangepast, was: True
    if do_bb_stability:
        print('**** Performing black-box lipschitz estimation over subset of dataset ****')
        maxpoints = 20
        #valid_loader 0 it's shuffled, so it's like doing random choice
        mini_test = next(iter(valid_loader))[0][:maxpoints].numpy()
        lips = expl.estimate_dataset_lipschitz(mini_test,
                                           n_jobs=-1, bound_type='box_std',
                                           eps=args.lip_eps, optim=args.optim,
                                           n_calls=args.lip_calls, verbose=2)
        pdb.set_trace()
        Stability_dict = {'lips': lips}
        pickle.dump(Stability_dict, open(results_path + '_stability_blackbox.pkl', "wb"))
        All_Results['stability_blackbox'] = lips 


    # add concept plot
    concept_grid(model, test_loader, top_k = 10, save_path = results_path + '/concept_grid.pdf')

    pickle.dump(All_Results, open(results_path + '_combined_metrics.pkl', "wb")) # Aangepast: .pkl was .format(dataname)

    


    # args.epoch_stats = epoch_stats
    # save_path = args.results_path
    # print("Save train/dev results to", save_path)
    # args_dict = vars(args)
    # pickle.dump(args_dict, open(save_path,'wb') )

if __name__ == '__main__':
    main()
