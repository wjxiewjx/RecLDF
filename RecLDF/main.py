import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color

from RecLDF import RecLDF
from data.dataset import RecLDFDataset

#from recbole.trainer import Trainer
from recbole.trainer.trainer import Trainer

from tqdm import tqdm
import torch.cuda.amp as amp
from recbole.utils import (
    set_color,
    get_gpu_usage,
)
from torch.nn.utils.clip_grad import clip_grad_norm_

class NewTrainer(Trainer):

    def __init__(self, config, model):
        super(NewTrainer, self).__init__(config, model)
        self.clip_grad_norm = True
        self.optimizer_supervised = torch.optim.Adam(self.model.parameters(), lr=config['lr_supervised'], weight_decay=1e-5)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        total_loss = 0
       
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )
        if (epoch_idx < 2) or (epoch_idx % 3 == 0):
            for batch_idx, interaction in enumerate(iter_data): #iter_data
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss_supervised(interaction)
                self._check_nan(loss)
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    #clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                self.optimizer_supervised.step()
                total_loss += loss.item()
        else:
            for batch_idx, interaction in enumerate(iter_data): #iter_data
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss(interaction)
                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )

        return total_loss


def finetune(dataset, pretrained_file, fix_enc=True, **kwargs):
    # configurations initialization
    props = ['props/RecLDF.yaml', 'props/finetune.yaml']
    print(props)

    # configurations initialization
    config = Config(model=RecLDF, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = RecLDFDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = RecLDF(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            logger.info(f'Fix encoder parameters.')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False
    logger.info(model)

    # trainer loading and initialization
    #trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    trainer = NewTrainer(config, model)
    #trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)


    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='OR', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('-f', type=bool, default=True)
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(args.d, pretrained_file=args.p, fix_enc=args.f)
