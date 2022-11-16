
from trainer import Trainer
from data import self_Dataset, self_DataLoader

import os
import time
import random
import numpy as np
from argument import print_args, parser
from utils import create_logger, mkdir

def main(args):
    if args.todo == 'train' or args.todo == 'valid':
        folder_name = '%dway_%dshot_%s_%s' % (args.nway, args.shots, args.model_type, args.affix)
        model_folder = os.path.join(args.model_root, folder_name)
        log_folder = os.path.join(args.log_root, folder_name)

        mkdir(args.model_root)
        mkdir(args.log_root)
        mkdir(model_folder)
        mkdir(log_folder)
        setattr(args, 'model_folder', model_folder)
        setattr(args, 'log_folder', log_folder)
        logger = create_logger(log_folder, args.todo)
        print_args(args, logger)

        tr_dataloader = self_DataLoader(args.data_root, 
            train=True, dataset=args.dataset, seed=args.seed, nway=args.nway)

        trainer_dict = {'args': args, 'logger': logger, 'tr_dataloader': tr_dataloader}

        trainer = Trainer(trainer_dict)


        if args.load:
            model_path = os.path.join(args.load_dir, 'model.pth')
            trainer.load_model(model_path)

        ###########################################
        ## start training

        trainer.train()



if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu
    main(args)
