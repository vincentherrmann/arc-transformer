import os
from collections import OrderedDict
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
from six import string_types
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl

from arc_transformer.dataset import ArcDataset, task_map, task_reduce
from arc_transformer.transformer_model import ArcTransformer
from arc_transformer.preprocessing import PriorCalculation
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
        self.prior_calc = PriorCalculation()
        self.model = ArcTransformer(feature_dim=128, feedforward_dim=512, num_priors=self.prior_calc.num_priors)

    def setup_datasets(self):
        self.training_set = ArcDataset(self.hparams.training_set_path)
        print("training set length:", len(self.training_set))
        self.validation_set = ArcDataset(self.hparams.validation_set_path)
        print("validation set length:", len(self.validation_set))


    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, task_data):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        task_data = task_map(task_data, lambda t: t[0])
        priors = task_map(task_data, self.prior_calc)
        task_data = task_map(task_data, lambda t: t + 1)
        task_data = self.make_uniform_size(task_data)
        priors = self.make_uniform_size(priors)

        self.model(task_data, priors)
        return None

    @staticmethod
    def make_uniform_size(data):
        size = task_reduce(data,
                           f=lambda r, t: [max(r[0], t.shape[0]), max(r[1], t.shape[1])],
                           start_value=[0, 0])
        def pad_end(t):
            padding = (0, size[1] - t.shape[1], 0, size[0] - t.shape[0])
            if len(t.shape) == 3:
                padding = (0, 0) + padding
            r = F.pad(t, pad=padding)
            return r
        data = task_map(data,
                        f=pad_end)
        return data

    def loss(self, scores):
        batch_size = scores.shape[0]

        # scores: data_batch, data_step, target_batch, target_step
        if self.hparams.score_over_all_timesteps:
            n_scores = scores.view(-1, batch_size, self.prediction_steps)  # data_batch*data_step, target_batch. target_step
            noise_scoring = torch.logsumexp(n_scores, dim=0)  # target_batch, target_step
            valid_scores = torch.diagonal(scores, dim1=0, dim2=2)  # data_step, target_step, batch
            valid_scores = torch.diagonal(valid_scores, dim1=0, dim2=1)  # batch, step
        else:
            scores = torch.diagonal(scores, dim1=1, dim2=3)  # data_batch, target_batch, step
            noise_scoring = torch.logsumexp(scores, dim=0)  # target_batch, target_step
            valid_scores = torch.diagonal(scores, dim1=0, dim2=1).permute([1, 0])  # batch, step

        prediction_losses = -torch.mean(valid_scores - noise_scoring, dim=1)
        loss = torch.mean(prediction_losses)

        return loss

    def training_step(self, batch, batch_i):
        """
        Lightning calls this inside the training loop
        :param data_batch:
        :return:
        """

        batch = batch[0]

        self.eval()

        # forward pass
        predicted_z, targets, _, _, scal = self.forward(batch)
        scores = torch.tensordot(predicted_z, targets, dims=([2], [1]))  # data_batch, data_step, target_batch, target_step
        loss = self.loss(scores)

        if self.hparams.wasserstein_penalty != 0.:
            score_sum = torch.sum(scores)
            scal_grad = torch.autograd.grad(outputs=score_sum,
                                             inputs=scal,
                                             create_graph=True,
                                             retain_graph=True,
                                             only_inputs=True)
            gradient_penalty = ((scal_grad[0].norm(2, dim=1) - 1) ** 2).mean()
            loss += self.hparams.wasserstein_penalty * gradient_penalty

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss = loss.unsqueeze(0)

        output = OrderedDict({
            'loss': loss,
            'progress_bar': {'train_loss': loss},
            'log': {'tng_loss': loss, 'lr': self.current_learning_rate}
        })

        return output

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        lr_scale = self.lr_scheduler.lr_lambda(self.global_step)
        for pg in optimizer.param_groups:
            self.current_learning_rate = lr_scale * self.hparams.learning_rate
            pg['lr'] = self.current_learning_rate

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def validation_step(self, data, batch_i):
        """
        Lightning calls this inside the validation loop
        :param data_batch:
        :return:
        """

        # forward pass
        result = self.forward(data)
        # predicted_z: batch, step, features
        # targets: batch, features, step
        scores = torch.tensordot(predicted_z, targets, dims=([2], [1]))  # data_batch, data_step, target_batch, target_step
        loss = self.loss(scores)  # valid scores: batch, time_step

        if False: #self.hparams.wasserstein_penalty != 0.:
            score_sum = torch.sum(scores)
            score_sum.requires_grad = True
            scal.requires_grad = True
            scal_grad = torch.autograd.grad(outputs=score_sum,
                                             inputs=scal,
                                             create_graph=True,
                                             retain_graph=True,
                                             only_inputs=True,
                                             allow_unused=True)
            gradient_penalty = ((scal_grad[0].norm(2, dim=1) - 1) ** 2).mean()
            loss += self.hparams.wasserstein_penalty * gradient_penalty

        # calculate prediction accuracy as the proportion of scores that are highest for the correct target
        prediction_template = torch.arange(0, scores.shape[0], dtype=torch.long)
        if self.hparams.score_over_all_timesteps:
            prediction_template = torch.arange(0, scores.shape[0]*scores.shape[1], dtype=torch.long, device=scores.device)
            prediction_template = prediction_template.view(scores.shape[0], scores.shape[1])
            max_score = torch.argmax(scores.view(scores.shape[0], scores.shape[1], -1), dim=2)  # batch, step
        else:
            scores = torch.diagonal(scores, dim1=1, dim2=3)  # data_batch, target_batch, step
            prediction_template = torch.arange(0, scores.shape[0], dtype=torch.long, device=scores.device)
            prediction_template = prediction_template[:, None].repeat(1, self.hparams.prediction_steps)
            max_score = torch.argmax(scores, dim=1) # batch, step

        correctly_predicted = torch.eq(prediction_template, max_score)
        prediction_accuracy = torch.sum(correctly_predicted).float() / prediction_template.numel() # per prediction step  # self.validation_examples_per_batch

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss = loss.unsqueeze(0)
            prediction_accuracy = prediction_accuracy.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss,
            'val_acc': torch.mean(prediction_accuracy)
        })

        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

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

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        test_task_dict = {}
        if self.get_test_task_model is not None:
            test_task_model = self.get_test_task_model()
            test_task_model.calculate_data()

            # if not self.experiment.debug:
            #     self.experiment.add_embedding(self.test_task_model.task_data,
            #                                   metadata=self.test_task_model.task_labels,
            #                                   global_step=self.global_step)

            trainer = pl.Trainer(logger=False,
                                 checkpoint_callback=False,
                                 max_epochs=100)
            trainer.fit(test_task_model)

            test_task_dict['val_task_acc'] = trainer.callback_metrics['val_accuracy']
            test_task_dict['val_task_loss'] = trainer.callback_metrics['val_loss']

        tqdm_dic = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {
            "progress_bar": tqdm_dic,
            "log": {"val_loss": val_loss,
                    "val_acc": val_acc_mean,
                    "val_task_acc": trainer.callback_metrics['val_accuracy'],
                    "val_task_loss": trainer.callback_metrics['val_loss']},
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
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.lr_scheduler = WarmupCosineSchedule(optimizer=optimizer,
                                                 warmup_steps=self.hparams.warmup_steps,
                                                 t_total=self.hparams.annealing_steps)
        self.current_learning_rate = 0.
        return optimizer

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
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        return loader

    @pl.data_loader
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
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        return loader

    @pl.data_loader
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