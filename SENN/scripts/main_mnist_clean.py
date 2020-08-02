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

# Local imports
from SENN.arglist import get_senn_parser #parse_args as parse_senn_args
from SENN.models import GSENN
from SENN.conceptizers import image_fcc_conceptizer, image_cnn_conceptizer, input_conceptizer

from SENN.parametrizers import image_parametrizer
from SENN.aggregators import linear_scalar_aggregator, additive_scalar_aggregator
from SENN.trainers import HLearningClassTrainer, VanillaClassTrainer, GradPenaltyTrainer
from SENN.utils import plot_theta_stability, generate_dir_names, noise_stability_plots, concept_grid
from SENN.eval_utils import estimate_dataset_lipschitz

from robust_interpret.explainers import gsenn_wrapper
from robust_interpret.utils import lipschitz_boxplot, lipschitz_argmax_plot, lipschitz_feature_argmax_plot

from random import sample
from tqdm import tqdm
from collections import defaultdict


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
                        batch_size=args.batch_size, num_workers=args.num_workers
                        )

    # Set h_type 
    if args.h_type == 'input':
        conceptizer  = input_conceptizer()
        args.nconcepts = 28*28 + int(not args.nobias)
    elif args.h_type == 'cnn':        
        conceptizer  = image_cnn_conceptizer(28*28, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)
    else:
        conceptizer  = image_fcc_conceptizer(28*28, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)

    # Initialize model
    parametrizer = image_parametrizer(28*28, args.nconcepts, args.theta_dim,  only_positive = args.positive_theta)
    aggregator   = additive_scalar_aggregator(args.concept_dim, args.nclasses)
    model        = GSENN(conceptizer, parametrizer, aggregator)

    # If load_model == True, load existing model
    if args.load_model:
    	checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
    	checkpoint.keys()
    	model = checkpoint['model']

    # Specify theta regression type
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

    # Train model
    if not args.load_model and args.train:
        trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)
        trainer.plot_losses(save_path=results_path)
    
    # If nothing is specified, load model and use VanillaClassTrainer
    else:
        checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']
        trainer =  VanillaClassTrainer(model, args)


    print("Done training/ loading model")

    # Evaluation
    ### 1. Single point lipshiz estimate via black box optim
    # All methods tested with BB optim for fair comparison)
    features = None
    classes = [str(i) for i in range(10)]
    model.eval()
    expl = gsenn_wrapper(model,
                        mode      = 'classification',
                        input_type = 'image',
                        multiclass=True,
                        feature_names = features,
                        class_names   = classes,
                        train_data      = train_loader,
                        skip_bias = True,
                        verbose = False)

    # Make noise stability plots, (Figure 4 paper)
    print("Results_path", results_path)
    # noise_stability_plots(model, test_tds, cuda = args.cuda, save_path = results_path)

    # One sample
    # plt.imshow(test_tds[1][0].reshape(28,28))
    # plt.show()

    print(test_tds)

    distances = defaultdict(list)

    print(distances)

    scale = 0.5

    print(distances)

    for i in tqdm(range(1000)):

	    x = Variable(test_tds[i][0].view(1,1,28,28), volatile = True)

	    true_class = test_tds[i][1][0].item()

	    pred = model(x)

	    theta = model.thetas.data.cpu().numpy().squeeze()

	    klass = pred.data.max(1)[1]
	    deps = theta[:,klass].squeeze()

	    # print("prediction", klass)
	    # print("dependencies", deps)

	    # Add noise to sample and repeat
	    noise = Variable(scale*torch.randn(x.size()), volatile = True)

	    pred = model(noise)

	    theta = model.thetas.data.cpu().numpy().squeeze()

	    klass_noise = pred.data.max(1)[1]
	    deps_noise = theta[:,klass].squeeze()

	    dist = np.linalg.norm(deps - deps_noise)

	    distances[true_class].append(dist)

    print(distances)

if __name__ == '__main__':
	main()

