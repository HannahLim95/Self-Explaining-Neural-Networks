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
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
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
from robust_interpret.utils import lipschitz_boxplot, lipschitz_argmax_plot

# Download data
from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples
EMNIST = list_datasets()
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.list_IDs[index], self.labels[index]

def load_emnist_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size = 64,
                    num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    # images_train, labels_train = extract_training_samples('letters')
    # images_test, labels_test = extract_test_samples('letters')
    images_train, labels_train = extract_training_samples('balanced')
    images_test, labels_test = extract_test_samples('balanced')
    # images_train = images_train[:256]
    # labels_train = labels_train[:256]

    train = Dataset(images_train, labels_train)
    test = Dataset(images_test, labels_test)
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
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)

    train_loader = dataloader.DataLoader(train, sampler=train_sampler, **dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    return train_loader, valid_loader, test_loader, train, test

def parse_args():
    senn_parser = get_senn_parser()

    ### Local ones
    parser = argparse.ArgumentParser(parents =[senn_parser],add_help=False,
        description='Interpretability robustness evaluation on EMNIST')

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
    args.nclasses = 47  #26 for 'letters' dataset
    args.h_type = 'cnn'
    args.theta_dim = args.nclasses

    model_path, log_path, results_path = generate_dir_names('emnist', args)

    load_emnist_data(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    train_loader, valid_loader, test_loader, train_tds, test_tds = load_emnist_data(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # args.h_type == 'cnn':
    conceptizer = image_cnn_conceptizer(28 * 28, args.nconcepts, args.concept_dim)  # , sparsity = sparsity_l)

    parametrizer = image_parametrizer(28*28, args.nconcepts, args.theta_dim,  only_positive = args.positive_theta)

    aggregator   = additive_scalar_aggregator(args.concept_dim, args.nclasses)

    model        = GSENN(conceptizer, parametrizer, aggregator) #, learn_h = args.train_h)
    if args.load_model:   #false
        checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']

    # args.theta_reg_type == 'grad3'
    trainer = GradPenaltyTrainer(model, args, typ=3)

    if not args.load_model and args.train:

        trainer.train(train_loader, valid_loader, epochs=args.epochs, save_path=model_path)
        trainer.plot_losses(save_path=results_path)
    else:
        checkpoint = torch.load(os.path.join(model_path, 'model_best.pth.tar'),
                                map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']
        trainer = VanillaClassTrainer(model, args)
    trainer.validate(test_loader, fold='test')

    All_Results = {}

    ### 1. Single point lipshiz estimate via black box optim
    # All methods tested with BB optim for fair comparison)
    features = None
    classes = [str(i) for i in range(10)]
    model.eval()
    expl = gsenn_wrapper(model,
                         mode='classification',
                         input_type='image',
                         multiclass=True,
                         feature_names=features,
                         class_names=classes,
                         train_data=train_loader,
                         skip_bias=True,
                         verbose=False)

    #### Debug single input
    # x = next(iter(train_tds))[0]
    # attr = expl(x, show_plot = False)
    # pdb.set_trace()

    # #### Debug multi input
    # x = next(iter(test_loader))[0] # Transformed
    # x_raw = test_loader.dataset.test_data[:args.batch_size,:,:]
    # attr = expl(x, x_raw = x_raw, show_plot = True)
    # #pdb.set_trace()

    # #### Debug argmax plot_theta_stability
    # if args.h_type == 'input':
    #     x = next(iter(test_tds))[0].numpy()
    #     y = next(iter(test_tds))[0].numpy()
    #     x_raw = (test_tds.test_data[0].float() / 255).numpy()
    #     y_raw = revert_to_raw(x)
    #     att_x = expl(x, show_plot=False)
    #     att_y = expl(y, show_plot=False)
    #     lip = 1
    #     lipschitz_argmax_plot(x_raw, y_raw, att_x, att_y, lip)  # save_path=fpath)
    #     # pdb.set_trace()

    # ### 2. Single example lipschitz estimate with Black Box
    # do_bb_stability_example = True
    # if do_bb_stability_example:
    #     print('**** Performing lipschitz estimation for a single point ****')
    #
    #     idx = 0
    #     print('Example index: {}'.format(idx))
    #     # x = train_tds[idx][0].view(1,28,28).numpy()
    #     # x = next(iter(test_tds))[0].numpy()
    #     x = next(iter(test_tds))[0]  #removed numpy
    #     print(test_tds) #<__main__.Dataset object at 0x14bfa6208>   #WIJ HEBBEN ANDERE SOORT DATASET
    #     x_raw = (test_tds.test_data[0].float() / 255).numpy()
    #     # x_raw = next(iter(train_tds))[0]
    #
    #     # args.optim     = 'gp'
    #     # args.lip_eps   = 0.1
    #     # args.lip_calls = 10
    #     Results = {}
    #
    #     lip, argmax = expl.local_lipschitz_estimate(x, bound_type='box_std',
    #                                                 optim=args.optim,
    #                                                 eps=args.lip_eps,
    #                                                 n_calls=4 * args.lip_calls,
    #                                                 njobs=1,
    #                                                 verbose=2)
    #     # pdb.set_trace()
    #     Results['lip_argmax'] = (x, argmax, lip)
    #     # .reshape(inputs.shape[0], inputs.shape[1], -1)
    #     att = expl(x, None, show_plot=False)  # .squeeze()
    #     # .reshape(inputs.shape[0], inputs.shape[1], -1)
    #     att_argmax = expl(argmax, None, show_plot=False)  # .squeeze()
    #
    #     # pdb.set_trace()
    #     Argmax_dict = {'lip': lip, 'argmax': argmax, 'x': x}
    #     fpath = os.path.join(results_path, 'argmax_lip_gp_senn.pdf')
    #     if args.h_type == 'input':
    #         lipschitz_argmax_plot(x_raw, revert_to_raw(argmax), att, att_argmax, lip, save_path=fpath)
    #     pickle.dump(Argmax_dict, open(
    #         results_path + '/argmax_lip_gp_senn.pkl', "wb"))
    #     pdb.set_trace()
    #     # print(asd.asd)

    # noise_stability_plots(model, test_tds, cuda = args.cuda, save_path = results_path)
    ### 3. Local lipschitz estimate over multiple samples with Black BOx Optim
    # do_bb_stability = False
    # if do_bb_stability:
    #     print('**** Performing black-box lipschitz estimation over subset of dataset ****')
    #     maxpoints = 20
    #     # valid_loader 0 it's shuffled, so it's like doing random choice
    #     mini_test = next(iter(valid_loader))[0][:maxpoints].numpy()
    #     lips = expl.estimate_dataset_lipschitz(mini_test,
    #                                            n_jobs=-1, bound_type='box_std',
    #                                            eps=args.lip_eps, optim=args.optim,
    #                                            n_calls=args.lip_calls, verbose=2)
    #     pdb.set_trace()
    #     Stability_dict = {'lips': lips}
    #     pickle.dump(Stability_dict, open(results_path + '_stability_blackbox.pkl', "wb"))
    #     All_Results['stability_blackbox'] = lips

        # add concept plot
    concept_grid(model, test_loader, top_k=10, save_path=results_path + '/concept_grid.pdf')

    # pickle.dump(All_Results, open(results_path + '_combined_metrics.pkl'.format(dataname), "wb"))

    # args.epoch_stats = epoch_stats
    # save_path = args.results_path
    # print("Save train/dev results to", save_path)
    # args_dict = vars(args)
    # pickle.dump(args_dict, open(save_path,'wb') )

if __name__ == '__main__':
    main()