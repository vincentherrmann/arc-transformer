import os
from collections import OrderedDict
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn
from six import string_types
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from arc_transformer.dataset import ArcDataset, task_map, task_reduce
from arc_transformer.transformer_model import ArcTransformer
from arc_transformer.preprocessing import Preprocessing, PositionalEncoding
from arc_transformer.lr_schedulers import WarmupCosineSchedule


class ArcSystem(pl.LightningModule):
    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.setup_datasets()
        self.prior_calc = Preprocessing()
        self.positional_encoding = PositionalEncoding(features_dim=64)
        self.generator = ArcTransformer(input_dim=11, feature_dim=64, feedforward_dim=256, output_dim=11,
                                        relative_priors_dim=self.prior_calc.num_positional_features,
                                        num_iterations=3)
        self.discriminator = ArcTransformer(input_dim=11, feature_dim=64, feedforward_dim=256, output_dim=1,
                                            relative_priors_dim=self.prior_calc.num_positional_features,
                                            num_iterations=2)
        self.gumbel_temp = 5.
        self.generated_grids = None
        self.generated_priors = None

    def setup_datasets(self):
        self.training_set = ArcDataset(self.hparams.training_set_path, max_size=(20, 20), augment=True)
        print("training set length:", len(self.training_set))
        self.validation_set = ArcDataset(self.hparams.validation_set_path, max_size=(20, 20), augment=False)
        print("validation set length:", len(self.validation_set))

    def preprocessing(self, task_data):
        task_data = task_map(task_data, lambda t: self.prior_calc(t))
        relative_priors = task_map(task_data, lambda t: self.prior_calc.get_relative_priors(t))
        task_data, max_size = self.make_uniform_size(task_data, (20, 20))
        relative_priors, _ = self.make_uniform_size(relative_priors, (39, 39))

        #task_map(task_data, lambda t: print("grid u shape", t.shape))
        #task_map(priors, lambda t: print("prior u shape", t.shape))

        return task_data, relative_priors

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, task_data, relative_priors):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        grids, pair_features, task_features = self.generator(task_data, relative_priors)
        test_out_grid = grids[:, -1]
        return test_out_grid

    # @staticmethod
    # def make_uniform_size(data, size=None):
    #     if size is None:
    #         size = task_reduce(data,
    #                            f=lambda r, t: [max(r[0], t.shape[0]), max(r[1], t.shape[1])],
    #                            start_value=[0, 0])
    #     def pad_end(t):
    #         padding = (0, 0, 0, size[1] - t.shape[1], 0, size[0] - t.shape[0])
    #         if len(t.shape) == 4:
    #             padding = (0, 0) + padding
    #         r = F.pad(t, pad=padding)
    #         return r
    #     data = task_map(data,
    #                     f=pad_end)
    #     return data, size

    # def on_after_backward(self):
    #     parameters = self.named_parameters()
    #     for name, p in parameters:
    #         if p.grad is None:
    #             print("no grad for", name)
    #         elif (p.grad != p.grad).any():
    #             print("nan in gradient of", name)
    #         #else:
    #         #    print("regular grad for", name)
    #     pass

    def adversarial_loss(self, grid_rating, y):
        y_hat = grid_rating.mean(dim=1)
        return F.binary_cross_entropy_with_logits(y_hat, y)
        #return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """
        Lightning calls this inside the training loop
        :param data_batch:
        :return:
        """
        data, priors = batch
        batch_size = data.shape[0]

        # train generator
        if optimizer_idx == 0:
            target_solutions = data[:, -1].argmax(dim=3)
            generator_data = data.clone()
            generator_data[:, -1] = 1.

            #generator_priors = priors.clone()
            test_out_priors = self.training_set.preprocessing.get_relative_priors(generator_data[:, -1, :, :, 0])
            priors[:, -1] = test_out_priors
            generator_priors = priors.clone()

            generator_output = self.forward(generator_data, generator_priors)

            # !!! only for testing TODO delete
            #generator_output[:, 0] += 5.

            generated_grids = F.gumbel_softmax(generator_output, tau=self.gumbel_temp, hard=True)
            generated_grids = generated_grids.view(batch_size, data.shape[2], data.shape[3], data.shape[4])
            generated_solution = generated_grids.argmax(dim=3)

            correct_cells = ((generated_solution - target_solutions[0]) == 0).sum().item()
            accuracy = correct_cells / (generated_solution.shape[1] * generated_solution.shape[2])
            print("accuracy:", accuracy * 100, "%")

            num_valid_cells = (target_solutions != 0).sum().item()
            correct_valid_cells = ((generated_solution[target_solutions != 0] - \
                                    target_solutions[target_solutions != 0]) == 0).sum().item()
            valid_accuracy = correct_valid_cells / num_valid_cells
            print("valid accuracy:", valid_accuracy * 100, "%")

            # remove rows and columns that have no valid values and calculate priors for the generated grids
            # test_out_priors = []
            # for valid_grid in generated_grids:
            #     valid_rows = valid_grid[..., 0].prod(dim=1, keepdim=True) == 0.
            #     valid_cols = valid_grid[..., 0].prod(dim=0, keepdim=True) == 0.
            #     valid_grid = valid_grid[valid_rows * valid_cols].view(valid_rows.sum().item(),
            #                                                           valid_cols.sum().item(),
            #                                                           valid_grid.shape[2])
            #     gp = self.training_set.preprocessing.get_relative_priors(valid_grid[..., 0])
            #     test_out_priors.append(gp)
            # test_out_priors = torch.stack(test_out_priors, dim=0)
            #
            # generator_data[:, -1] = generated_grids
            # generator_priors = generator_priors.clone()
            # generator_priors[:, -1] = test_out_priors
            #
            # self.generated_grids = generator_data
            # self.generated_priors = generator_priors
            #
            # grid_rating, _, _ = self.discriminator(self.generated_grids, self.generated_priors)
            # grid_rating = grid_rating[:, -1, 0].view(batch_size, -1)
            # valid = torch.ones_like(grid_rating[:, 0])
            # g_loss = self.adversarial_loss(grid_rating, valid)
            #
            # ce_loss = F.cross_entropy(generator_output, target_solutions.view(-1))
            #ce_loss = F.cross_entropy(generated_grids.view(-1, 11), target_solutions.view(-1))
            g_loss = ce_loss #+ g_loss

            # print("generator loss:", g_loss.item())

            tqdm_dict = {'g_loss': g_loss, 'accuracy': accuracy, 'valid_accuracy': valid_accuracy}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            grid_rating, _, _ = self.discriminator(data, priors)
            grid_rating = grid_rating[:, -1, 0].view(batch_size, -1)
            valid = torch.ones_like(grid_rating[:, 0])
            real_loss = self.adversarial_loss(grid_rating, valid)

            # how well can it label as fake?
            grid_rating, _, _ = self.discriminator(self.generated_grids.detach(), self.generated_priors.detach())
            grid_rating = grid_rating[:, -1, 0].view(batch_size, -1)
            fake = torch.zeros_like(grid_rating[:, 0])
            fake_loss = self.adversarial_loss(grid_rating, fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            # print("discriminator loss:", d_loss.item())

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        super().optimizer_step(current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure)
        current_step = self.trainer.global_step
        if current_step % 50 == 0:
            self.parameter_schedules(current_step)

    def parameter_schedules(self, step):
        self.gumbel_temp *= 0.9
        print("gumbel temp:", self.gumbel_temp)

    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    #     lr_scale = self.lr_scheduler.lr_lambda(self.global_step)
    #     for pg in optimizer.param_groups:
    #         self.current_learning_rate = lr_scale * self.hparams.learning_rate
    #         pg['lr'] = self.current_learning_rate
    #
    #     # update params
    #     optimizer.step()
    #     optimizer.zero_grad()

    # def validation_step(self, data, batch_i):
    #     """
    #     Lightning calls this inside the validation loop
    #     :param data_batch:
    #     :return:
    #     """
    #
    #     # forward pass
    #     # predicted_z: batch, step, features
    #     # targets: batch, features, step
    #
    #     task_data, priors = self.preprocessing(data)
    #     target = task_data["test"][-1]["output"].clone()
    #     mask = (target.sum(dim=2) == 0.).long()
    #     target = (mask * 10) + target.argmax(dim=2) * (1-mask)
    #     task_data["test"][-1]["output"] *= 0.
    #     task_data["test"][-1]["output"][..., 0] = 1.
    #
    #     test_grid_out = self.forward(task_data, priors)
    #
    #     prediction = test_grid_out.argmax(dim=1).view(target.shape[0], target.shape[1])
    #     acc = (target.flatten() == prediction.flatten()).sum().float() / target.numel()
    #     loss = F.cross_entropy(test_grid_out.view(-1, 11), target.view(-1))
    #
    #     # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
    #     if self.trainer.use_dp:
    #         loss = loss.unsqueeze(0)
    #
    #     output = OrderedDict({
    #         'val_loss': loss,
    #         'val_acc': acc
    #     })
    #
    #     return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss = 0
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        l = len(outputs)
        l = l if l > 0 else 1

        val_loss_mean /= l
        val_acc_mean /= l

        tqdm_dic = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {
            "progress_bar": tqdm_dic,
            "log": {"val_loss": val_loss,
                    "val_acc": val_acc_mean},
            "val_loss": val_loss_mean
        }
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate)

        # self.lr_scheduler = WarmupCosineSchedule(optimizer=opt_g,
        #                                          warmup_steps=self.hparams.warmup_steps,
        #                                          t_total=self.hparams.annealing_steps)
        # self.current_learning_rate = 0.
        return [opt_g], []
        #return [opt_g, opt_d], []

    #@pl.data_loader
    def train_dataloader(self):
        print('tng data loader called')
        if torch.cuda.is_available():
            torch.cuda.seed()
        else:
            torch.seed()
        loader = DataLoader(
            dataset=self.training_set,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
        return loader

    #@pl.data_loader
    def val_dataloader(self):
        print('val data loader called')
        if torch.cuda.is_available():
            torch.cuda.manual_seed(12345)
        else:
            torch.manual_seed(12345)
        loader = DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
        return loader

    #@pl.data_loader
    def test_dataloader(self):
        print('test data loader called')
        return None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])

        parser.add_argument('--log_dir', default='logs', type=str)
        parser.add_argument('--checkpoint_dir', default='checkpoints', type=str)

        # data
        parser.add_argument('--training_set_path', default='data/training', type=str)
        parser.add_argument('--validation_set_path', default='data/validation', type=str)
        parser.add_argument('--test_task_set_path', default='data/test_task', type=str)

        # training
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--learning_rate', default=0.0001, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--use_all_GPUs', default=True, type=bool)
        parser.add_argument('--warmup_steps', default=0, type=int)
        parser.add_argument('--annealing_steps', default=1e6, type=int)

        return parser