from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.logging import TensorBoardLogger
import os.path
import sys
import argparse

from arc_transformer.arc_system import ArcSystem


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data-path', metavar='DIR', type=str,
                               help='path to dataset')
    parent_parser.add_argument('--save-path', metavar='DIR', default=".", type=str,
                               help='path to save output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')

    parser = ArcSystem.add_model_specific_args(parent_parser)
    return parser.parse_args()

def main(hparams, cluster=None, results_dict=None):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # init experiment

    name = "arc_adversarial_test_12_wasserstein_w_clipping_overfit"
    logs_dir = "../../logs"
    checkpoint_dir = "../../checkpoints/" + name
    hparams.training_set_path = '../../data/training'
    hparams.validation_set_path = '../../data/evaluation'
    hparams.test_task_set_path = '../../data/test'
    hparams.data_path = hparams.training_set_path

    hparams.num_heads = 4
    hparams.max_grid_height = 12
    hparams.max_grid_width = 12
    hparams.augment = False
    hparams.weight_clip_value = 0.1

    hparams.learning_rate = 0.0001

    hparams.batch_size = 1

    # build model
    model = ArcSystem(hparams)

    logger = TensorBoardLogger(save_dir=logs_dir, name=name)
    logger = False

    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_dir, save_top_k=1)
    checkpoint_callback = False
    # configure trainer
    #torch.autograd.set_detect_anomaly(True)
    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                      train_percent_check=0.02,
                      val_percent_check=0.3,
                      #val_check_interval=1.,
                      logger=logger,
                      checkpoint_callback=checkpoint_callback,
                      fast_dev_run=False,
                      early_stop_callback=False,
                      precision=32,
                      accumulate_grad_batches=1,
                      print_nan_grads=True,
                      num_sanity_val_steps=3)

    # train model
    trainer.fit(model)

if __name__ == '__main__':
    hyperparams = get_args()

    # train model
    main(hyperparams)