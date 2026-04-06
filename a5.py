# Disclaimer: each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use and applicable links before the script runs and the data is placed in the user machine.

import argparse, sys, time, os
import pdb
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn
from utils.utils import *
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor, BoundDataParallel, CrossEntropyWrapper
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import BoundExp
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
from auto_LiRPA.utils import MultiAverageMeter
from tensorboardX import SummaryWriter
from models.robustifier import zx2x_rob, xx_rob2z
from autoattack import AutoAttack

def parse_argumments():
  # parse argument
  parser = argparse.ArgumentParser()

  # tasks
  parser.add_argument('--train-prototypes', action='store_true', help='train prototypes')
  parser.add_argument('--train-robustifier', action='store_true', help='train robustifier')
  parser.add_argument('--train-classifier', action='store_true', help='train classifier')
  parser.add_argument('--test', action='store_true', help='test')
  parser.add_argument('--no-autoattack', action='store_true', help='do not use autoattack for testing (faster)')
  parser.add_argument('--attack-type', type=str, default='autoattack', 
                     choices=['autoattack', 'pgd', 'fgsm', 'genattack'], 
                     help='type of attack to use during testing')
  parser.add_argument('--pgd-steps', type=int, default=40,
                     help='number of steps for PGD attack')
  parser.add_argument('--pgd-alpha', type=float, default=0.01,
                     help='step size for PGD attack')
  # archs
  parser.add_argument('--robustifier-arch', type=str, choices=['mnist', 'cifar10', 'tinyimagenet', 'identity', 'path'], default='mnist', help='robustifier architecture')
  parser.add_argument('--acquisition-arch', type=str, choices=['identity', 'camera'], default='identity', help='acquisition device architecture')
  parser.add_argument('--classifier-arch', type=str, choices=['mnist', 'cifar10', 'tinyimagenet', 'fonts','path'], default='mnist', help='classifier architecture')

  # dataset
  parser.add_argument('--training-dataset-folder', type=str, default=None, help='training dataset folder (default: None)')
  parser.add_argument('--validation-dataset-folder', type=str, default=None, help='validation dataset folder (default: None)')
  parser.add_argument('--test-dataset-folder', type=str, default=None, help='testing dataset folder (default: None)')
  parser.add_argument('--prototypes-dataset-folder', type=str, default=None, help='dataset with trained prototypes (default: None)')
  parser.add_argument('--multi-norm-training', action='store_true', help='Train against L1, L2, and Linf simultaneously')
  parser.add_argument('--l2-eps-multiplier', type=float, default=10.0, help='Multiplier for L2 epsilon relative to Linf')
  parser.add_argument('--l1-eps-multiplier', type=float, default=100.0, help='Multiplier for L1 epsilon relative to Linf')
  # training params
  parser.add_argument('--batch-size', type=int, default=100, help='batch size (default 100)')
  parser.add_argument('--epochs', type=int, default=100, help='training epochs (default 100)')
  parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default 1e-3)')
  parser.add_argument("--lr-scheduler-milestones", type=int, nargs='+', default=[25, 42], help='list of epoch milestones to decrease lr (default [25, 42])')
  parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='gamma for lr scheduler (default 0.1)')
  parser.add_argument("--x-epsilon-attack-scheduler-name", type=str, default="SmoothedScheduler", choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler", "FixedScheduler"], help='epsilon attack x scheduler (default SmoothScheduler)')
  parser.add_argument("--x-epsilon-attack-scheduler-opts", type=str, default='start=3,length=18,mid=0.3', help='options for epsilon attack x scheduler (default ''start=3,length=18,mid=0.3''')
  parser.add_argument("--x-augmentation-mnist", action='store_true', help='augment x during training, for mnist')
  parser.add_argument("--x-augmentation-cifar10", action='store_true', help='augment x during training, for cifar')
  parser.add_argument("--save-interval", type=int, default=5, help="interval for saving model (in epochs)")
  parser.add_argument('--visualize', action='store_true', help='Generate plot for 2 successes and 2 fails')
  parser.add_argument("--batch-multiplier", type=int, default=1, help='batch multiplicative factor (reduces memory consumption) - default: 1')
  parser.add_argument("--test-multiplier", type=int, default=1, help='test multiplicative factore (reduces the variance of test) - default: 1' )

  # load
  parser.add_argument('--load-classifier', type=str, default=None, help='if provided, load the classifier indicated here')
  parser.add_argument('--load-robustifier', type=str, default=None, help='if provided, load the robustifier indicated here')

  # log and save
  parser.add_argument('--log-dir', type=str, default='log/', help='log folder')
  #parser.add_argument('--save-w-ratio', type=float, default=1.0, help='ratio of the modified dataset to be saved in the log dir (can be used as dataset in future - also save debug images)')

  # attack params
  parser.add_argument('--x-epsilon-attack-training', type=float, default=0.1, help='epsilon for MitM attack during training')
  parser.add_argument('--x-epsilon-attack-testing', type=float, default=0.1, help='epsilon for MitM attack during testing')
  parser.add_argument('--w-epsilon-attack-training', type=float, default=0.0, help='epsilon for physical attack during training')
  parser.add_argument('--w-epsilon-attack-testing', type=float, default=0.0, help='epsilon for physical attack during testing')

  # robustifier
  parser.add_argument('--x-epsilon-defense', type=float, default=0.1, help='epsilon for defense (x)')
  parser.add_argument('--w-epsilon-defense', type=float, default=0.0, help='epsilon for defense (w)')

  # auto_LiRPA parameters
  parser.add_argument("--bound-type", type=str, default="CROWN-IBP", choices=["IBP", "CROWN-IBP", "CROWN", "CROWN-FAST"], help='method of bound analysis')

  # verbose
  parser.add_argument('--verbose', action='store_true', help='verbose')
  # norm
  parser.add_argument('--attack-norm', type=float, default=np.inf, choices=[1.0, 2.0, np.inf], help='Lp norm to use for attacks and perturbations (1.0 for L1, 2.0 for L2, np.inf for L-inf)')
  args = parser.parse_args()

  return args

def fgsm_attack(model, x, y, epsilon, norm, device="cuda"):
    """
    Fast Gradient Sign Method
    """
    x.requires_grad = True
    
    # Forward pass
    outputs = model(x)
    loss = torch.nn.CrossEntropyLoss()(outputs, y)
    
    # Backward pass
    loss.backward()
    
    # Create the perturbed image
    if norm == np.inf:
        perturbed_image = x + epsilon * x.grad.sign()
    elif norm == 2.0:
        grad_norm = torch.norm(x.grad.view(x.size(0), -1), p=2, dim=1).view(-1, 1, 1, 1)
        scaled_grad = x.grad / (grad_norm + 1e-10)
        perturbed_image = x + epsilon * scaled_grad
    elif norm == 1.0:
        grad_norm = torch.norm(x.grad.view(x.size(0), -1), p=1, dim=1).view(-1, 1, 1, 1)
        scaled_grad = x.grad / (grad_norm + 1e-10)
        perturbed_image = x + epsilon * scaled_grad
    else:
        raise NotImplementedError(f"Norm {norm} not implemented for FGSM")
    
    # Project back to valid range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image.detach()


# PGD attack implementation
def pgd_attack(model, x, y, epsilon, alpha, num_steps, norm, device="cuda"):
    """
    Projected Gradient Descent Attack
    """
    # Create copy of the input
    x_adv = x.clone().detach()
    
    # Random initialization within epsilon ball if using more than 1 step
    if num_steps > 1:
        if norm == np.inf:
            x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-epsilon, epsilon)
        elif norm == 2.0:
            delta = torch.zeros_like(x_adv).normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta = delta * r * epsilon / (n + 1e-8)
            x_adv = x_adv + delta
        elif norm == 1.0:
            # For L1 norm, random sparse perturbation
            delta = torch.zeros_like(x_adv)
            for i in range(x_adv.size(0)):
                idx = torch.randint(0, x_adv[i].numel(), (1,)).item()
                delta_flat = delta[i].view(-1)
                delta_flat[idx] = torch.sign(torch.randn(1)).item() * epsilon
                delta[i] = delta_flat.view_as(delta[i])
            x_adv = x_adv + delta
    
    # Clamp to valid image range
    x_adv = torch.clamp(x_adv, 0, 1)
    
    # PGD iterations
    for _ in range(num_steps):
        x_adv.requires_grad = True
        
        # Forward pass
        outputs = model(x_adv)
        loss = torch.nn.CrossEntropyLoss()(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient step
        with torch.no_grad():
            if norm == np.inf:
                x_adv = x_adv + alpha * x_adv.grad.sign()
            elif norm == 2.0:
                grad_norm = torch.norm(x_adv.grad.view(x_adv.size(0), -1), p=2, dim=1).view(-1, 1, 1, 1)
                scaled_grad = x_adv.grad / (grad_norm + 1e-10)
                x_adv = x_adv + alpha * scaled_grad
            elif norm == 1.0:
                grad_flat = x_adv.grad.view(x_adv.size(0), -1)
                abs_grad_flat = torch.abs(grad_flat)
                # Get indices of max absolute gradient for each sample
                _, indices = torch.max(abs_grad_flat, dim=1)
                # Create sparse gradient with only the max component
                sparse_grad = torch.zeros_like(grad_flat)
                for i in range(x_adv.size(0)):
                    sparse_grad[i, indices[i]] = torch.sign(grad_flat[i, indices[i]])
                # Reshape back to image dimensions
                sparse_grad = sparse_grad.view_as(x_adv)
                x_adv = x_adv + alpha * sparse_grad
            
            # Project back to epsilon ball around original image
            if norm == np.inf:
                x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
            elif norm == 2.0:
                delta = x_adv - x
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
                mask = n > epsilon
                d_flat = delta.view(delta.size(0), -1)
                d_flat[mask.squeeze()] = d_flat[mask.squeeze()] * epsilon / n[mask]
                delta = d_flat.view_as(delta)
                x_adv = x + delta
            elif norm == 1.0:
                delta = x_adv - x
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=1, dim=1).view(-1, 1, 1, 1)
                mask = n > epsilon
                if mask.any():
                    # Project to L1 ball - approximation by scaling
                    d_flat = delta.view(delta.size(0), -1)
                    d_flat[mask.squeeze()] = d_flat[mask.squeeze()] * epsilon / n[mask]
                    delta = d_flat.view_as(delta)
                x_adv = x + delta
                
            # Clamp to valid image range
            x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


# GenAttack implementation (simplified version)
class GenAttack:
    def __init__(self, model, norm, eps, pop_size=6, mutation_rate=0.15, alpha=0.15, iterations=100):
        self.model = model
        self.norm = norm
        self.eps = eps
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.iterations = iterations
        
    def attack(self, x, y, device="cuda"):
        batch_size = x.size(0)
        # Initialize population
        population = [x.clone()]
        for _ in range(self.pop_size - 1):
            if self.norm == np.inf:
                delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
            elif self.norm == 2.0:
                delta = torch.zeros_like(x).normal_()
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
                delta = delta * self.eps / (n + 1e-8)
            elif self.norm == 1.0:
                delta = torch.zeros_like(x)
                d_flat = delta.view(delta.size(0), -1)
                # Sparse perturbation for L1
                idx = torch.randint(0, d_flat.size(1), (batch_size, int(d_flat.size(1) * 0.01)))
                for i in range(batch_size):
                    d_flat[i, idx[i]] = torch.sign(torch.randn(idx.size(1))) * self.eps / idx.size(1)
                delta = d_flat.view_as(x)
            
            perturbed = torch.clamp(x + delta, 0, 1)
            population.append(perturbed)
        
        # Evolutionary optimization
        for _ in range(self.iterations):
            # Evaluate fitness
            fitness = []
            for p in population:
                with torch.no_grad():
                    outputs = self.model(p)
                    # Negative CE loss as fitness (higher is better)
                    loss = -torch.nn.functional.cross_entropy(outputs, y, reduction='none')
                    fitness.append(loss)
            
            # Convert to tensor
            fitness = torch.stack(fitness, dim=1)
            
            # Get best candidates per batch element
            _, indices = torch.topk(fitness, k=2, dim=1)
            
            # Create new population
            new_population = [population[0]]  # Keep original
            
            # Elite
            elite = torch.zeros_like(x)
            for i in range(batch_size):
                elite[i] = population[indices[i, 0]][i]
            new_population.append(elite)
            
            # Generate new members through crossover and mutation
            for _ in range(self.pop_size - 2):
                parent1 = torch.zeros_like(x)
                parent2 = torch.zeros_like(x)
                
                for i in range(batch_size):
                    parent1[i] = population[indices[i, 0]][i]  # Best parent
                    parent2[i] = population[indices[i, 1]][i]  # Second best parent
                
                # Crossover
                beta = torch.zeros_like(x).uniform_(0, 1)
                child = beta * parent1 + (1 - beta) * parent2
                
                # Mutation
                mask = torch.zeros_like(x).uniform_(0, 1) < self.mutation_rate
                mutation = torch.zeros_like(x)
                
                if self.norm == np.inf:
                    mutation[mask] = torch.zeros_like(mutation[mask]).uniform_(-self.alpha, self.alpha)
                elif self.norm == 2.0:
                    mutation = torch.zeros_like(x).normal_() * self.alpha
                elif self.norm == 1.0:
                    # Sparse mutation
                    for i in range(batch_size):
                        flat_mask = mask[i].flatten()
                        if flat_mask.sum() > 0:
                            idx = torch.where(flat_mask)[0]
                            mutation_flat = mutation[i].flatten()
                            mutation_flat[idx] = torch.sign(torch.randn(idx.size(0))) * self.alpha
                            mutation[i] = mutation_flat.view_as(mutation[i])
                
                child = child + mutation
                
                # Project to epsilon neighborhood
                delta = child - x
                if self.norm == np.inf:
                    delta = torch.clamp(delta, -self.eps, self.eps)
                elif self.norm == 2.0:
                    d_flat = delta.view(delta.size(0), -1)
                    n = d_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
                    mask = n > self.eps
                    if mask.any():
                        d_flat[mask.squeeze()] = d_flat[mask.squeeze()] * self.eps / n[mask]
                    delta = d_flat.view_as(delta)
                elif self.norm == 1.0:
                    d_flat = delta.view(delta.size(0), -1)
                    n = d_flat.norm(p=1, dim=1).view(-1, 1, 1, 1)
                    mask = n > self.eps
                    if mask.any():
                        d_flat[mask.squeeze()] = d_flat[mask.squeeze()] * self.eps / n[mask]
                    delta = d_flat.view_as(delta)
                
                child = torch.clamp(x + delta, 0, 1)
                new_population.append(child)
            
            population = new_population
        
        # Return best adversarial example
        with torch.no_grad():
            fitness = []
            for p in population:
                outputs = self.model(p)
                loss = -torch.nn.functional.cross_entropy(outputs, y, reduction='none')
                fitness.append(loss)
            
            fitness = torch.stack(fitness, dim=1)
            _, indices = torch.max(fitness, dim=1)
            
            best_adv = torch.zeros_like(x)
            for i in range(batch_size):
                best_adv[i] = population[indices[i]][i]
            
        return best_adv

def compute_predictions_and_loss(classifier, normalized_x, normalized_x_min, normalized_x_max, normalized_x_epsilon, f, bound_type, y, num_classes, ce, attack_norm):
    # Quick compute for easy case
  if f == 1:
    prediction = classifier(normalized_x)  # natural prediction
    reg_ce = ce(prediction, y)
    reg_err = torch.sum(torch.argmax(prediction, dim=1) != y).cpu().detach().numpy() / y.size(0)
    ver_ce = reg_ce
    ver_err = reg_err
    loss = reg_ce
    return (prediction, reg_ce, reg_err, ver_ce, ver_err, loss)

  # prediction: lower and upper bounds (auto_LiRPA) - also use the linear comb in the last layer
  scalar_eps = normalized_x_epsilon.max() if isinstance(normalized_x_epsilon, torch.Tensor) else normalized_x_epsilon
  
  ptb = PerturbationLpNorm(
        norm=attack_norm,  
        eps=scalar_eps, 
        x_L=torch.max(normalized_x - normalized_x_epsilon.view(1, -1, 1, 1), normalized_x_min.view(1, -1, 1, 1)), 
        x_U=torch.min(normalized_x + normalized_x_epsilon.view(1, -1, 1, 1), normalized_x_max.view(1, -1, 1, 1))
    )
  data = BoundedTensor(normalized_x, ptb)

  c = torch.eye(num_classes).type_as(data)[y].unsqueeze(1) - torch.eye(num_classes).type_as(data).unsqueeze(0).cuda()
  I = (~(y.data.unsqueeze(1) == torch.arange(num_classes).type_as(y.data).unsqueeze(0))).cuda()
  c = (c[I].view(data.size(0), num_classes - 1, num_classes)).cuda()

  # prediction: clean data
  prediction = classifier(data)  # natural prediction
  reg_ce = ce(prediction, y)
  reg_err = torch.sum(torch.argmax(prediction, dim=1) != y).cpu().detach().numpy() / y.size(0)

  if bound_type == 'CROWN-IBP':
    ilb, iub = classifier.compute_bounds(IBP=True, C=c, method=None)
    if f < 1e-5:
      lb = ilb
    else:
      clb, cub = classifier.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
      lb = clb * f + ilb * (1 - f)
  else:
    lb, ub = classifier.compute_bounds(x=(data,), method=bound_type, C=c)

  lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
  fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
  ver_ce = ce(-lb_padded, fake_labels)
  ver_err = torch.sum((lb < 0).any(dim=1)).item() / data.size(0)

  # loss
  loss = reg_ce * f + ver_ce * (1.0 - f)

  return (prediction, reg_ce, reg_err, ver_ce, ver_err, loss)


def vprint(s, verbose):
  if verbose:
    print(s)


def main():

  ###################################################################################################
  # Parser

  args = parse_argumments()
  logger = LogDir(args.log_dir)
  logger.parser(sys, args)

  ###################################################################################################
  # Prepare dataset

  # TODO load only the usefull datasets
  (train_loader, train_avg, train_std, train_num, train_siz, train_num_samples, train_width, train_height, train_num_channels, train_names) = open_dataset(args.training_dataset_folder, args.batch_size)
  (validation_loader, validation_avg, validation_std, validation_num, validation_siz, validation_validation_num_samples, validation_width, validation_height, validation_num_channels, validation_names) = open_dataset(args.validation_dataset_folder, args.batch_size, shuffle=False)
  (test_loader, test_avg, test_std, test_num, test_siz, test_num_samples, test_width, test_height, test_num_channels, test_names) = open_dataset(args.test_dataset_folder, args.batch_size, shuffle=False)
  (prototypes_loader, prototypes_avg, prototypes_std, prototypes_num, prototypes_siz, prototypes_num_samples, prototypes_width, prototypes_height, prototypes_num_channels, prototypes_names) = open_dataset(args.prototypes_dataset_folder, args.batch_size, shuffle=False)
  if not(train_loader is None):
    avg, std, num, width, height, num_channels, names = train_avg, train_std, train_num, train_width, train_height, train_num_channels, train_names
  elif not(validation_loader is None):
    avg, std, num, width, height, num_channels, names = validation_avg, validation_std, validation_num, validation_width, validation_height, validation_num_channels, validation_names
  elif not(test_loader is None):
    avg, std, num,  width, height, num_channels, names = test_avg, test_std, test_num, test_width, test_height, test_num_channels, test_names
  normalized_x_min = (0.0 - avg) / std
  normalized_x_max = (1.0 - avg) / std
  vprint("Created datasets and loader. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 **3), torch.cuda.max_memory_allocated() / (1024 **3), torch.cuda.memory_reserved() / (1024 **3), torch.cuda.max_memory_reserved() / (1024 **3)), args.verbose)

  # normalize
  normalize = transforms.Normalize(avg, std)

  ###################################################################################################
  # Prepare and load models

  # models
  if args.acquisition_arch == 'identity':
    tr1 = None
    tr2 = None
    def acquisition(x, tr1, tr2):
      return x
  elif args.acquisition_arch == 'camera':
    tr1 = transforms.Compose([transforms.RandomCrop(128, 5, fill=0.0),
                             transforms.RandomRotation(5, fill=0.0),
                             transforms.RandomPerspective(distortion_scale=0.25, p=0.99, interpolation=2, fill=0.0)])
    tr2 = transforms.Compose([transforms.GaussianBlur(kernel_size=9, sigma=(0.01, 1.0)),
                             transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)])

    def acquisition(x, tr1, tr2):
      y = torch.clamp(tr2((1.0 - tr1(x) + torch.randn(x.size()).cuda() * (0.001 + 0.099 * torch.rand(1).cuda())).expand(-1, 3, -1, -1)), min=0.0, max=1.0)
      return y

  (robustifier_ori, classifier_ori) = create_models(robustifier_arch=args.robustifier_arch, x_min=0.0, x_max=1.0, x_avg=avg, x_std=std, x_epsilon_defense=args.x_epsilon_defense, robustifier_filename=args.load_robustifier, classifier_arch=args.classifier_arch, classifier_filename=args.load_classifier)
  dummy_input = acquisition(torch.zeros((2, num_channels, height, width)).cuda(), tr1, tr2)
  robustifier = robustifier_ori.cuda() # no need to use a bounded model! # BoundedModule(robustifier_ori, dummy_input, bound_opts={'relu': 'same-slope', 'conv_mode': 'patches'}, device='cuda:0')
  classifier = BoundedModule(classifier_ori, dummy_input, bound_opts={'relu': 'same-slope', 'conv_mode': 'patches'}, device='cuda')


  if args.x_augmentation_mnist:
    augmentation = transforms.Compose([transforms.RandomCrop(28, 4), transforms.RandomRotation(10)]) # TODO this works for MNIST only
  elif args.x_augmentation_cifar10:
    augmentation = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)])  # TODO this works for CIFAR10 only
  else:
    augmentation = torch.nn.Sequential()

  vprint("Created models. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 **3), torch.cuda.max_memory_allocated() / (1024 **3), torch.cuda.memory_reserved() / (1024 **3), torch.cuda.max_memory_reserved() / (1024 **3)), args.verbose)

  # TODO when training to defende against a physical attack, we must consider the camera + robustifier + classifier as a unique classifier

  # init
  ce = torch.nn.CrossEntropyLoss()
  attack_norm = args.attack_norm
  ###################################################################################################
  # Train

  # TODO solve the ResourceWarning: unclosed file <_io.BufferedReader name='./webdataset_mnist/train/data-000003.tar'> yield sample warning!

  if (args.train_prototypes) or (args.train_robustifier) or (args.train_classifier):

    # z
    z = torch.zeros((train_num_samples, num_channels, height, width), dtype=torch.float32, requires_grad=True, device='cuda')

    # optimizer
    parameters_list = []
    if args.train_prototypes:
      parameters_list = parameters_list + [z, ]
    if args.train_robustifier:
      parameters_list = parameters_list + list(robustifier_ori.parameters())
    if args.train_classifier:
      parameters_list = parameters_list + list(classifier_ori.parameters())
    opt = torch.optim.RMSprop(parameters_list, lr=args.lr)
    opt.zero_grad()

    # schedulers
    x_epsilon_attack_scheduler = eval(args.x_epsilon_attack_scheduler_name)(args.x_epsilon_attack_training, args.x_epsilon_attack_scheduler_opts.replace(" ", ""))
    x_epsilon_attack_scheduler.set_epoch_length(int(train_num_samples / args.batch_size))
    x_epsilon_attack_scheduler.step_batch()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_scheduler_milestones, gamma=args.lr_scheduler_gamma, verbose=False)

    # tensorboard
    writer = SummaryWriter(logger.tb_dir)

    # init
    t0 = time.time()
    freq = 0
    global_step = 0
    best_ver_err = 1.0
    x_epsilon_attack_scheduler.train()
    if not args.train_classifier:
        classifier.eval()
    else:
        classifier.train()

    if not args.train_robustifier:
        robustifier.eval()
    else:
        robustifier.train()
    batch_multiplier = args.batch_multiplier
    batch_multiplier_index = 0

    # epochs
    for epoch in range(args.epochs):

      # go through the entire dataset
      if not(train_loader is None):
        for (i, (w, y, idxs)) in enumerate(train_loader):

          # to GPU
          w = w.cuda().float()
          y = y.cuda()
          idxs = idxs.cuda()
          vprint("Loaded batch data w. Memory [allocated %.3fGb [max %.3fGb] reserved %.3fGb [max %.3fGb]." % (torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.max_memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)), args.verbose)

# Setup Multi-Norm parameters
          x_epsilon_attack_base = x_epsilon_attack_scheduler.get_eps()
          f = (x_epsilon_attack_scheduler.get_max_eps() - x_epsilon_attack_base) / np.max((x_epsilon_attack_scheduler.get_max_eps(), 1e-12))

          # Determine the active norms
          active_norms = [np.inf, 2.0, 1.0] if args.multi_norm_training else [args.attack_norm]
          
          # ROUND-ROBIN: Select ONE norm for this specific batch based on the batch index (i)
          current_norm = active_norms[i % len(active_norms)]

          # Rebuild the forward passes
          w_rob = zx2x_rob(z=torch.gather(z, 0, idxs.view(-1, 1, 1, 1).expand(args.batch_size, num_channels, height, width)), x=w, x_epsilon= torch.tensor([args.w_epsilon_defense]).float().cuda(), xD_min=torch.tensor([0.0]).float().cuda(), xD_max=torch.tensor([1.0]).float().cuda())
          x = augmentation(acquisition(w_rob, tr1, tr2))
          normalized_x = normalize(x)
          normalized_x_rob = robustifier(normalized_x)
          normalized_x_rob = torch.max(torch.min(normalized_x_rob, normalized_x_max.view(1, -1, 1, 1)), normalized_x_min.view(1, -1, 1, 1))

          # Scale epsilon based on the chosen norm
          if current_norm == np.inf:
              x_epsilon_attack = x_epsilon_attack_base
          elif current_norm == 2.0:
              x_epsilon_attack = x_epsilon_attack_base * args.l2_eps_multiplier
          elif current_norm == 1.0:
              x_epsilon_attack = x_epsilon_attack_base * args.l1_eps_multiplier

          normalized_x_epsilon_attack = x_epsilon_attack / std

          (prediction, reg_ce, reg_err, ver_ce, ver_err, loss) = compute_predictions_and_loss(
              classifier=classifier,
              normalized_x=normalized_x_rob,
              normalized_x_min=normalized_x_min,
              normalized_x_max=normalized_x_max,
              normalized_x_epsilon=normalized_x_epsilon_attack,
              f=f,
              bound_type=args.bound_type,
              y=y,
              num_classes=num.numpy()[0],
              ce=ce,
              attack_norm=current_norm
          )

          vprint(f"Classified batch {i} with norm {current_norm}. Memory [allocated {torch.cuda.memory_allocated() / (1024 ** 3):.3f}Gb]", args.verbose)

          # Standard single backward pass (Memory Safe!)
          loss.backward()
          vprint("Backward done.", args.verbose)

          # Step the optimizer
          batch_multiplier_index += 1
          if np.mod(batch_multiplier_index, batch_multiplier) == 0:
            opt.step()
            opt.zero_grad()
            batch_multiplier_index = 0
          # step info
          with torch.no_grad():
            psnr_w = 20.0 * torch.log10(1.0 / torch.sqrt(((w_rob - w) ** 2.0 + 1e-12).mean()))
            x_rob = (normalized_x_rob * std.view(1, -1, 1, 1)) + avg.view(1, -1, 1, 1)
            psnr_x = 20.0 * torch.log10(1.0 / torch.sqrt((((x_rob - x) * std.view(1, -1, 1, 1))**2.0 + 1e-12).mean()))

            # tensorboard log
            writer.add_scalars('Loss', {'reg ce [training]': reg_ce, 'ver ce [training]': ver_ce, 'loss [training]': loss}, global_step=global_step)
            writer.add_scalars('Loss - f', {'f': f}, global_step=global_step)
            writer.add_scalars('Error', {'reg [training]': reg_err, 'ver [training]': ver_err}, global_step=global_step)
            writer.add_scalars('Epsilon x', {'[training]': x_epsilon_attack_base, '[testing]': args.x_epsilon_attack_testing}, global_step=global_step)
            writer.add_scalars('PSNR', {'w [training]': psnr_w, 'x [training]': psnr_x}, global_step=global_step)

            if np.mod(i, 10) == 0:
              print("Epoch %.4d batch %.5d eps %.5f f %.5f reg_ce %.5f [err %.2f%%] ver_ce %.5f [ver err %.2f%%] loss %.5f psnr x %.2fdB psnr w %.2fdB." % (epoch, i, x_epsilon_attack, f, reg_ce, reg_err*100.0, ver_ce, ver_err*100.0, loss, psnr_x, psnr_w))

          # updates after each step
          x_epsilon_attack_scheduler.step_batch(verbose=True)
          global_step += 1

          # timing (log only)
          t1 = time.time()
          freq = 0.99 * freq + 0.01 * 1.0 / (t1 - t0)
          if np.mod(i, 10) == 0:
            vprint("Iteration time: %.2fms [%.2fHz - filtered: %.2fHz]" % ((t1 - t0) * 1000.0, 1.0 / (t1 - t0), freq), args.verbose)
          t0 = time.time()

      # validation
      if not (validation_loader is None):
        meter = MultiAverageMeter()
        was_classifier_training = classifier.training
        was_robustifier_training = robustifier.training
        classifier.eval()
        robustifier.eval()
        with torch.no_grad():
          epsilon_x_attack = x_epsilon_attack_scheduler.get_eps()
          normalized_x_epsilon_attack = epsilon_x_attack / std
          for (i, (w, y, idxs)) in enumerate(validation_loader):
            w = w.cuda()
            y = y.cuda()
            idxs = idxs.cuda()
            w_rob = zx2x_rob(z=torch.gather(z, 0, idxs.view(-1, 1, 1, 1).expand(args.batch_size, num_channels, height, width)), x=w, x_epsilon=torch.tensor([args.w_epsilon_defense]).float().cuda(), xD_min=torch.tensor([0.0]).float().cuda(), xD_max=torch.tensor([1.0]).float().cuda())
            x = acquisition(w_rob, tr1, tr2)
            normalized_x = normalize(x)
            normalized_x_rob = robustifier(normalized_x)
            normalized_x_rob = torch.max(torch.min(normalized_x_rob, normalized_x_max.view(1, -1, 1, 1)), normalized_x_min.view(1, -1, 1, 1))
            (prediction, reg_ce, reg_err, ver_ce, ver_err, loss) = compute_predictions_and_loss(classifier=classifier,
                                                                                                normalized_x=normalized_x_rob,
                                                                                                normalized_x_min=normalized_x_min,
                                                                                                normalized_x_max=normalized_x_max,
                                                                                                normalized_x_epsilon=normalized_x_epsilon_attack,
                                                                                                f=f,
                                                                                                bound_type=args.bound_type,
                                                                                                y=y,
                                                                                                num_classes=num.numpy()[0],
                                                                                                ce=ce,
                                                                                                attack_norm=attack_norm)
            meter.update('reg_ce', reg_ce, args.batch_size)
            meter.update('reg_err', reg_err, args.batch_size)
            meter.update('ver_ce', ver_ce, args.batch_size)
            meter.update('ver_err', ver_err, args.batch_size)
            meter.update('loss', loss, args.batch_size)
            psnr_w = 20.0 * torch.log10(1.0 / torch.sqrt(((w_rob - w) ** 2.0 + 1e-12).mean()))
            x_rob = (normalized_x_rob * std.view(1, -1, 1, 1)) + avg.view(1, -1, 1, 1)
            psnr_x = 20.0 * torch.log10(1.0 / torch.sqrt((((x_rob - x) * std.view(1, -1, 1, 1))**2.0 + 1e-12).mean()))
            meter.update('psnr_x', psnr_x, args.batch_size)
            meter.update('psnr_w', psnr_w, args.batch_size)

            # save images in tensorboard for debugging
            if i == 0:
              if args.train_prototypes:
                for n in range(np.min([10, args.batch_size])):
                  img = torch.cat((w[n], torch.clamp(0.5 + (w_rob[n] - w[n]) / args.w_epsilon_defense, min=0.0, max=1.0), w_rob[n]), dim=2)
                  writer.add_image('Prototye %d' % (n), img, global_step=global_step)

          s = "Epoch %.4d [Validation] eps %.5f f %.5f reg_ce %.5f [err %.2f%%] ver_ce %.5f [ver err %.2f%%] loss %.5f psnr x %.2fdB psnr w %.2fdB." % (epoch, epsilon_x_attack, f, meter.avg('reg_ce'), meter.avg('reg_err') * 100.0, meter.avg('ver_ce'), meter.avg('ver_err') * 100.0, meter.avg('loss'), meter.avg('psnr_x'), meter.avg('psnr_w'))
          print(s)

        # Write validation log
        if epoch == 0:
          ff = open(os.path.join(logger.eval_dir, 'train_eval.txt'), 'w')
        else:
          ff = open(os.path.join(logger.eval_dir, 'train_eval.txt'), 'a')
        ff.write(s + '\n')
        ff.close()

        # tensorboard log (validation and lr)
        # tensorboard log
        writer.add_scalars('Loss', {'reg ce [training]': reg_ce, 'ver ce [training]': ver_ce, 'loss [training]': loss}, global_step=global_step)
        writer.add_scalars('Error', {'reg [validation]': meter.avg('reg_err'), 'ver [validation]': meter.avg('ver_err')}, global_step=global_step)
        writer.add_scalars('PSNR', {'w [validation]': meter.avg('psnr_w'), 'x [validation]': meter.avg('psnr_x')}, global_step=global_step)
        writer.add_scalars('LR', {'lr': lr_scheduler.get_last_lr()[0]}, global_step=global_step)

        # save model if best in evaluation
        if best_ver_err > meter.avg('ver_err'):
          best_ver_err = meter.avg('ver_err')
          if args.train_classifier:
            torch.save({'state_dict': classifier_ori.model.state_dict(), 'epoch': epoch}, logger.model_dir + '/classifier_best')
          if args.train_robustifier:
            torch.save({'state_dict': robustifier_ori.state_dict(), 'epoch': epoch}, logger.model_dir + '/robustifier_best')
        if was_classifier_training:
            classifier.train()
        if was_robustifier_training:
            robustifier.train()
      # save model at the end of each 5 epochs
      if np.mod(epoch + 1, args.save_interval) == 0:
        if args.train_classifier:
          torch.save({'state_dict': classifier_ori.model.state_dict(), 'epoch': epoch}, logger.model_dir + '/classifier_epoch_%.4d' % (epoch))
        if args.train_robustifier:
          torch.save({'state_dict': robustifier_ori.state_dict(), 'epoch': epoch}, logger.model_dir + '/robustifier_epoch_%.4d' % (epoch))

      # lr scheduler step
      lr_scheduler.step()

    # tensorboard
    writer.close()

  # save modified w at the end of training (if needed)
  if args.train_prototypes:

    base_pattern = logger.w_dir
    num_samples = int(train_num_samples)
    torch.save(avg, os.path.join(base_pattern, "m.pt"))
    torch.save(std, os.path.join(base_pattern, "s.pt"))
    torch.save(num, os.path.join(base_pattern, "n.pt"))
    torch.save(num_samples, os.path.join(base_pattern, "num_samples.pt"))
    torch.save(names, os.path.join(base_pattern, "names.pt"))

    # webdataset and images saving
    with torch.no_grad():
      pattern = os.path.join(base_pattern, f"data-%06d.tar")
      sink = wds.ShardWriter(pattern, maxcount=10000)
      for (i, data) in enumerate(train_loader.dataset.iterator()):
        idx = data[2]
        key = "%.6d" % idx
        w = data[0].cuda()
        w_rob = zx2x_rob(z=z[idx:idx+1], x=w, x_epsilon=torch.tensor([args.w_epsilon_defense]).float().cuda(), xD_min=torch.tensor([0.0]).float().cuda(), xD_max=torch.tensor([1.0]).float().cuda()).squeeze()
        if w_rob.size(0) == 3:
          w_rob = w_rob.permute(1, 2, 0)
        sample = {"__key__": key,
                  "ppm": (w_rob * 255).cpu().numpy().astype(np.uint8),
                  "cls": data[1],
                  "pyd": idx}
        sink.write(sample)
        if np.mod(i, 100):
          print("saving: %.2f%%." % (100.0 * i / num_samples))

        if i == 0:
          torch.save(np.shape(data[0]), os.path.join(base_pattern, "size.pt"))

      sink.close()

  ###################################################################################################
  # Test

  if args.test and not(test_loader is None):
        classifier.eval()
        robustifier.eval()
        forward_pass = torch.nn.Sequential(normalize, classifier_ori.model)
        
        # Determine which norms to test
        active_norms = [np.inf, 2.0, 1.0] if args.multi_norm_training else [args.attack_norm]
        norm_map = {1.0: 'L1', 2.0: 'L2', np.inf: 'Linf'}

        # We will loop through each norm and run a full evaluation
        for current_norm in active_norms:
            attack_norm_str = norm_map.get(current_norm, 'Linf')
            print(f"\n{'='*60}")
            print(f"STARTING COMPREHENSIVE EVALUATION FOR NORM: {attack_norm_str}")
            print(f"{'='*60}\n")

            # Scale testing epsilon based on the current norm being evaluated
            if current_norm == np.inf:
                x_epsilon_attack = args.x_epsilon_attack_testing
            elif current_norm == 2.0:
                x_epsilon_attack = args.x_epsilon_attack_testing * args.l2_eps_multiplier
            elif current_norm == 1.0:
                x_epsilon_attack = args.x_epsilon_attack_testing * args.l1_eps_multiplier

            normalized_x_epsilon_attack = x_epsilon_attack / std

            # Initialize attack for the specific norm
            if args.attack_type == 'autoattack':
                adversary = AutoAttack(
                    forward_pass, 
                    norm=attack_norm_str, 
                    eps=x_epsilon_attack, 
                    version='standard'
                )
            elif args.attack_type == 'genattack':
                adversary = GenAttack(
                    forward_pass,
                    norm=current_norm,
                    eps=x_epsilon_attack,
                    iterations=50
                )
        
            # Prepare z vectors for the test set
            z = torch.zeros((test_num_samples, num_channels, height, width), dtype=torch.float32, requires_grad=False, device='cuda')
            if not (prototypes_loader is None):
                for (i, (w, y, idxs)) in enumerate(test_loader):
                    for n in range(w.size(0)):
                        idx = idxs[n]
                        z[idx] = w[n]
                w_robs = z.clone()
                for (i, (w_rob, y, idxs)) in enumerate(prototypes_loader):
                    for n in range(w_rob.size(0)):
                        idx = idxs[n]
                        w_robs[idx] = w_rob[n]
                z = xx_rob2z(z, w_robs, args.w_epsilon_defense)
                del w_rob
            f = 0.0

            detailed_results = {
                'norm': attack_norm_str,
                'epsilon': x_epsilon_attack,
                'batches': [],
                'summary': {
                    'total_samples': 0, 'total_reg_err': 0, 'total_ver_err': 0,
                    'total_adv_err': 0, 'total_psnr_w': 0, 'total_psnr_x': 0
                }
            }

            meter = MultiAverageMeter()
            with torch.no_grad():
                
                test_log_path = os.path.join(logger.eval_dir, f'test_eval_{args.attack_type}_{attack_norm_str}.txt')
                with open(test_log_path, 'w') as test_log:
                    test_log.write(f"Testing Results: Attack={args.attack_type.upper()}, Norm={attack_norm_str}, Eps={x_epsilon_attack}\n")
                    test_log.write("=" * 60 + "\n\n")

                    for j in range(args.test_multiplier):
                        test_log.write(f"Test Multiplier Run {j+1}\n")
                        test_log.write("-" * 30 + "\n")

                        for (i, (w, y, idxs)) in enumerate(test_loader):
                            w = w.cuda()
                            y = y.cuda()
                            idxs = idxs.cuda()

                            w_rob = zx2x_rob(
                                z=torch.gather(z, 0, idxs.view(-1, 1, 1, 1).expand(args.batch_size, num_channels, height, width)), 
                                x=w, 
                                x_epsilon=torch.tensor([args.w_epsilon_defense]).float().cuda(), 
                                xD_min=torch.tensor([0.0]).float().cuda(), 
                                xD_max=torch.tensor([1.0]).float().cuda()
                            )
                            x = acquisition(w_rob, tr1, tr2)
                            normalized_x = normalize(x)
                            normalized_x_rob = robustifier(normalized_x)
                            normalized_x_rob = torch.max(torch.min(normalized_x_rob, normalized_x_max.view(1, -1, 1, 1)), normalized_x_min.view(1, -1, 1, 1))
                            
                            (prediction, reg_ce, reg_err, ver_ce, ver_err, loss) = compute_predictions_and_loss(
                                classifier=classifier,
                                normalized_x=normalized_x_rob,
                                normalized_x_min=normalized_x_min,
                                normalized_x_max=normalized_x_max,
                                normalized_x_epsilon=normalized_x_epsilon_attack,
                                f=f,
                                bound_type=args.bound_type,
                                y=y,
                                num_classes=num.numpy()[0],
                                ce=ce,
                                attack_norm=current_norm
                            )

                            psnr_w = 20.0 * torch.log10(1.0 / torch.sqrt(((w_rob - w) ** 2.0 + 1e-12).mean()))
                            x_rob = (normalized_x_rob * std.view(1, -1, 1, 1)) + avg.view(1, -1, 1, 1)
                            psnr_x = 20.0 * torch.log10(1.0 / torch.sqrt((((x_rob - x) * std.view(1, -1, 1, 1)) ** 2.0 + 1e-12).mean()))

                            if args.no_autoattack:
                                adv_err = np.nan
                            else:
                                if args.attack_type == 'autoattack':
                                    x_adv = adversary.run_standard_evaluation(x_rob, y, bs=args.batch_size)
                                elif args.attack_type == 'pgd':
                                    with torch.enable_grad():
                                        x_adv = pgd_attack(forward_pass, x_rob, y, epsilon=x_epsilon_attack, alpha=args.pgd_alpha, num_steps=args.pgd_steps, norm=current_norm)
                                elif args.attack_type == 'fgsm':
                                    with torch.enable_grad():
                                        x_adv = fgsm_attack(forward_pass, x_rob, y, epsilon=x_epsilon_attack, norm=current_norm)
                                elif args.attack_type == 'genattack':
                                    x_adv = adversary.attack(x_rob, y)
                                
                                adv_prediction = classifier_ori(normalize(x_adv))
                                adv_err = torch.sum(torch.argmax(adv_prediction, dim=1) != y).cpu().detach().numpy() / y.size(0)

                            batch_result = {
                                'batch_index': i, 'reg_ce': reg_ce.item(), 'reg_err': reg_err,
                                'ver_ce': ver_ce.item(), 'ver_err': ver_err, 'loss': loss.item(),
                                'psnr_w': psnr_w.item(), 'psnr_x': psnr_x.item(), 'adv_err': adv_err
                            }
                            detailed_results['batches'].append(batch_result)

                            detailed_results['summary']['total_samples'] += y.size(0)
                            detailed_results['summary']['total_reg_err'] += reg_err * y.size(0)
                            detailed_results['summary']['total_ver_err'] += ver_err * y.size(0)
                            detailed_results['summary']['total_adv_err'] += adv_err * y.size(0)
                            detailed_results['summary']['total_psnr_w'] += psnr_w.item() * y.size(0)
                            detailed_results['summary']['total_psnr_x'] += psnr_x.item() * y.size(0)

                            batch_log_str = (
                                f"Batch {i}: "
                                f"Reg Err: {reg_err*100:.2f}% | "
                                f"Ver Err: {ver_err*100:.2f}% | "
                                f"Adv Err: {adv_err*100:.2f}% | "
                                f"Loss: {loss.item():.4f}"
                            )
                            print(batch_log_str)
                            test_log.write(batch_log_str + "\n")

                    summary = detailed_results['summary']
                    avg_reg_err = summary['total_reg_err'] / summary['total_samples'] * 100
                    avg_ver_err = summary['total_ver_err'] / summary['total_samples'] * 100
                    avg_adv_err = summary['total_adv_err'] / summary['total_samples'] * 100

                    summary_log_str = (
                        f"\n--- {attack_norm_str} FINAL SUMMARY ---\n"
                        f"Epsilon Used: {x_epsilon_attack}\n"
                        f"Average Regular Error: {avg_reg_err:.2f}%\n"
                        f"Average Verifiable Error: {avg_ver_err:.2f}%\n"
                        f"Average Adversarial Error: {avg_adv_err:.2f}%\n"
                    )
                    print(summary_log_str)
                    test_log.write(summary_log_str + "\n\n")

                import json
                with open(os.path.join(logger.eval_dir, f'detailed_test_results_{attack_norm_str}.json'), 'w') as f:
                    json.dump(detailed_results, f, indent=2)
###################################################################################################
  # Multi-Norm Visualization Generation (Searching for "Perfect" Improvements)
  
  if args.visualize and not(test_loader is None):
      from autoattack import AutoAttack

      print("\n" + "="*60)
      print("SCANNING FOR 'PERFECT' IMPROVEMENTS (EMPIRICAL AUTOATTACK)...")
      print("="*60)

      classifier.eval()
      robustifier.eval()
      classifier_ori.eval() 

      z = torch.zeros((test_num_samples, num_channels, height, width), dtype=torch.float32, requires_grad=False, device='cuda')

      def format_img(tensor):
          arr = tensor.detach().cpu().squeeze().numpy()
          if len(arr.shape) == 3:  
              arr = np.transpose(arr, (1, 2, 0))
          return np.clip(arr, 0, 1)

      active_norms = [np.inf, 2.0, 1.0] if args.multi_norm_training else [args.attack_norm]
      norm_map = {1.0: 'L1', 2.0: 'L2', np.inf: 'Linf'}

      forward_pass = torch.nn.Sequential(normalize, classifier_ori.model)

      with torch.no_grad():
          for current_norm in active_norms:
              attack_norm_str = norm_map.get(current_norm, 'Linf')
              print(f"\n--- Generating visuals for {attack_norm_str} ---")

              if current_norm == np.inf:
                  x_epsilon_attack = args.x_epsilon_attack_testing
                  aa_norm = 'Linf'
              elif current_norm == 2.0:
                  x_epsilon_attack = args.x_epsilon_attack_testing * args.l2_eps_multiplier
                  aa_norm = 'L2'
              elif current_norm == 1.0:
                  x_epsilon_attack = args.x_epsilon_attack_testing * args.l1_eps_multiplier
                  aa_norm = 'L1'
              
              adversary = AutoAttack(forward_pass, norm=aa_norm, eps=x_epsilon_attack, version='standard')
              adversary.verbose = False 

              perfect_success_cases = [] # Base Failed -> Rob Succeeded
              backup_success_cases = []  # Both Succeeded
              fail_cases = []            # Rob Failed
              
              scans = 0
              max_scans = 1000 # Scan up to 1000 images to find those perfect improvements!

              for (i, (w, y, idxs)) in enumerate(test_loader):
                  if (len(perfect_success_cases) >= 2 and len(fail_cases) >= 2) or scans >= max_scans:
                      break

                  w = w.cuda()
                  y = y.cuda()
                  idxs = idxs.cuda()
                  bs = w.size(0)
                  scans += bs

                  # 1. Base Images Pipeline
                  x_base = acquisition(w, tr1, tr2)
                  norm_x_base = normalize(x_base)
                  base_logits = classifier_ori.model(norm_x_base)
                  base_preds = torch.argmax(base_logits, dim=1)
                  base_probs = F.softmax(base_logits, dim=1)

                  # 2. Robustified Images Pipeline
                  w_rob = zx2x_rob(
                      z=torch.gather(z, 0, idxs.view(-1, 1, 1, 1).expand(bs, num_channels, height, width)), 
                      x=w,
                      x_epsilon=torch.tensor([args.w_epsilon_defense]).float().cuda(),
                      xD_min=torch.tensor([0.0]).float().cuda(), xD_max=torch.tensor([1.0]).float().cuda()
                  )
                  x_rob_acq = acquisition(w_rob, tr1, tr2)
                  norm_x_rob = robustifier(normalize(x_rob_acq))
                  norm_x_rob = torch.max(torch.min(norm_x_rob, normalized_x_max.view(1, -1, 1, 1)), normalized_x_min.view(1, -1, 1, 1))
                  x_rob_pixels = (norm_x_rob * std.view(1, -1, 1, 1)) + avg.view(1, -1, 1, 1)
                  
                  rob_logits = classifier_ori.model(normalize(x_rob_pixels))
                  rob_preds = torch.argmax(rob_logits, dim=1)
                  rob_probs = F.softmax(rob_logits, dim=1)

                  # 3. AutoAttack the Base Image
                  with torch.enable_grad(): 
                      x_base_adv = adversary.run_standard_evaluation(x_base, y, bs=bs)
                  base_adv_logits = classifier_ori.model(normalize(x_base_adv))
                  base_adv_preds = torch.argmax(base_adv_logits, dim=1)

                  # 4. AutoAttack the Robustified Image
                  with torch.enable_grad():
                      x_rob_adv = adversary.run_standard_evaluation(x_rob_pixels, y, bs=bs)
                  rob_adv_logits = classifier_ori.model(normalize(x_rob_adv))
                  rob_adv_preds = torch.argmax(rob_adv_logits, dim=1)
                  rob_adv_probs = F.softmax(rob_adv_logits, dim=1)

                  # 5. Extract Cases based on Improvement Logic
                  for n in range(bs):
                      if len(perfect_success_cases) >= 2 and len(fail_cases) >= 2:
                          break
                          
                      base_failed = (base_adv_preds[n] != y[n]).item()
                      rob_succeeded = (rob_adv_preds[n] == y[n]).item()

                      case_data = {
                          'w_img': format_img(w[n]),
                          'x_rob_img': format_img(x_rob_pixels[n]),
                          'y': y[n].item(),
                          'base_clean_pred': base_preds[n].item(),
                          'base_clean_prob': base_probs[n, base_preds[n]].item() * 100,
                          'base_adv_pred': base_adv_preds[n].item(),
                          'rob_clean_pred': rob_preds[n].item(),
                          'rob_clean_prob': rob_probs[n, rob_preds[n]].item() * 100,
                          'rob_adv_pred': rob_adv_preds[n].item(),
                          'rob_adv_prob': rob_adv_probs[n, rob_adv_preds[n]].item() * 100
                      }

                      if rob_succeeded:
                          if base_failed and len(perfect_success_cases) < 2:
                              perfect_success_cases.append(case_data)
                              print(f"  > Found PERFECT Success! (Base Fooled -> Rob Saved! Label {y[n].item()})")
                          elif not base_failed and len(backup_success_cases) < 2:
                              backup_success_cases.append(case_data)
                      else:
                          if len(fail_cases) < 2:
                              fail_cases.append(case_data)
                              print(f"  > Found Empirical Fail (Label {y[n].item()} -> Fooled to {rob_adv_preds[n].item()})")

              # Fallback Logic: Pad with backup cases if we couldn't find 2 perfect ones
              final_success_cases = perfect_success_cases.copy()
              if len(final_success_cases) < 2:
                  needed = 2 - len(final_success_cases)
                  final_success_cases.extend(backup_success_cases[:needed])

              all_cases = final_success_cases + fail_cases
              
              if len(all_cases) == 0:
                  print(f"Could not find valid cases for {attack_norm_str}. Skipping plot.")
                  continue

              fig, axes = plt.subplots(len(all_cases), 3, figsize=(14, 4.5 * len(all_cases)))
              
              if len(all_cases) == 1:
                  axes = np.expand_dims(axes, axis=0)

              for r, case in enumerate(all_cases):
                  # Col 1: Base Image
                  ax = axes[r, 0]
                  ax.imshow(case['w_img'], cmap='gray' if num_channels==1 else None)
                  ax.set_title(f"Clean Pred: {case['base_clean_pred']} ({case['base_clean_prob']:.1f}%)\nAttacked Pred: {case['base_adv_pred']}", fontsize=12)
                  ax.axis('off')

                  # Col 2: Robustified Image
                  ax = axes[r, 1]
                  ax.imshow(case['x_rob_img'], cmap='gray' if num_channels==1 else None)
                  ax.set_title(f"Rob Pred: {case['rob_clean_pred']} ({case['rob_clean_prob']:.1f}%)\nAttacked Rob Pred: {case['rob_adv_pred']} ({case['rob_adv_prob']:.1f}%)", fontsize=12)
                  ax.axis('off')

                  # Col 3: Difference Heatmap
                  ax = axes[r, 2]
                  diff = case['x_rob_img'] - case['w_img']
                  
                  if len(diff.shape) == 3 and diff.shape[-1] == 3: 
                      diff_viz = np.mean(diff, axis=-1)
                  else:
                      diff_viz = diff

                  vmax = np.max(np.abs(diff_viz)) if np.max(np.abs(diff_viz)) > 0 else 1e-5
                  im3 = ax.imshow(diff_viz, cmap='bwr', vmin=-vmax, vmax=vmax)
                  ax.set_title(f"Robustification for Ground Label: {case['y']}", fontsize=12)
                  fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
                  ax.axis('off')

              plt.suptitle(f"{attack_norm_str} Empirical AutoAttack Evaluation", fontsize=16, y=1.02)
              plt.tight_layout()
              plot_path = os.path.join(logger.eval_dir, f'empirical_visualization_{attack_norm_str}.png')
              plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
              plt.close(fig) 
              print(f"SUCCESS! {attack_norm_str} Visualization saved to: {plot_path}")

      print("\nVisualizations complete! Exiting to skip the standard test loop.")
      sys.exit(0)

  ##################################################################################################################################################
  return 0


if __name__=="__main__":
  import warnings
  warnings.simplefilter('ignore')

  main()
