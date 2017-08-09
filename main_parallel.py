from argparse import ArgumentParser

import cv2
import os
import sys
import itertools

import yaml
from torch import nn, optim, cuda, torch
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from model.google_net import GoogleNet
import numpy as np
from tqdm import tqdm

from dataset import HistDataset
from model.utils import to_scalar
from postprocessing.evaluator import Evaluator
from utils import fit_image, argmax, get_path

MAX_IMG_SIZE = (512, 512)


def batchify(samples):
    img_size = MAX_IMG_SIZE
    images = [fit_image(sample.image, img_size) for sample in samples if sample]
    images = np.stack(images, axis=0)
    labels = [sample.label for sample in samples if sample]
    return images, labels


def train_a_epoch(name, data, model, optimizer, criterion, cfg, en):
    evaluator = Evaluator(name, label2id=cfg['LABEL_2_ID'], pure_labels=cfg['PURE_LABELS'])
    print("evaluator loaded")
    i = 0
    for images, labels in tqdm(data, desc=name, total=np.math.ceil(len(data))):
        # zero the parameter gradients
        sys.stdout.flush()
        optimizer.zero_grad()
        model.zero_grad()
        # image = crop(sample.image, cfg.MAX_IMG_SIZE)
        inp = Variable(torch.from_numpy(images).float().cuda())
        inp = torch.transpose(inp, 1, 3)

        seq_out = model(inp)

        pred = argmax(seq_out)

        loss = criterion(seq_out, Variable(cuda.LongTensor(labels)))

        evaluator.append_data(to_scalar(loss), pred, labels)
        loss.backward()
        optimizer.step()
        i += 1

    print("Training Done")
    evaluator.gen_results()
    evaluator.print_results()
    evaluator.write_results(get_path(cfg['TRAIN_RESULT_FILE']),
                            "epoch = {2}; GOOGLE NET; lr={0}; batch_size={1}".format(cfg['LEARNING_RATE'],
                                                                                     cfg['BATCH_SIZE'],
                                                                                     en))

    return model


def test_a_epoch(name, data, model, result_file, cfg, en):
    pred_list = []
    true_list = []
    evaluator = Evaluator(name, label2id=cfg['LABEL_2_ID'], pure_labels=cfg['PURE_LABELS'])

    for images, labels in tqdm(data, desc=name, total=np.math.ceil(len(data))):
        inp = Variable(torch.from_numpy(images).float().cuda())
        inp = torch.transpose(inp, 1, 3)
        seq_out = model(inp)

        pred = argmax(seq_out)

        evaluator.append_data(0, pred, labels)

        # print(predicted)
        pred_list.extend(pred)
        true_list.extend(labels)

    evaluator.gen_results()
    evaluator.print_results()
    evaluator.write_results(result_file,
                            "epoch = {2}; GOOGLE NET; lr={0}; batch_size={1}".format(cfg['LEARNING_RATE'],
                                                                                     cfg['BATCH_SIZE'],
                                                                                     en))

    return evaluator, pred_list, true_list


def build_model(dataset, cfg):
    # init model
    model = GoogleNet(True, n_labels=cfg['NUM_LABELS'])

    print("Model Loaded")
    # Turn on cuda
    model = DataParallel(model, device_ids=cfg['DEVICE_IDS'])

    print("Model loaded in cuda memory")

    # verify model
    print(model)
    # print(list(model.parameters()))

    # init gradient descent optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=cfg['LEARNING_RATE'], weight_decay=cfg['L2_REG'])
    # optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9)
    optimizer.zero_grad()
    model.zero_grad()

    # init loss criteria
    criterion = nn.CrossEntropyLoss()

    best_res_val = 0.0
    best_epoch = 0
    for epoch in range(cfg['MAX_EPOCH']):
        print('-' * 40)
        print("EPOCH = {0}".format(epoch))
        print('-' * 40)

        train_loader = DataLoader(dataset.train, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['NUM_WORKERS'],
                                  collate_fn=batchify)

        print("train_loader ready")
        model = train_a_epoch("train", train_loader, model, optimizer, criterion, cfg, en=epoch)

        dev_loader = DataLoader(dataset.dev, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['NUM_WORKERS'],
                                collate_fn=batchify)

        dev_eval, pred_list, true_list = test_a_epoch("dev", dev_loader, model,
                                                      get_path(cfg['DEV_RESULT_FILE']), cfg, epoch)

        dev_eval.verify_results()

        if epoch == 0 or dev_eval.avg_fscore > best_res_val:
            best_epoch = epoch
            best_res_val = dev_eval.avg_fscore
            torch.save(model, get_path(cfg['MODEL_SAVE_FILE']))

        print("current dev score: {0}".format(dev_eval.avg_fscore))
        print("best dev score: {0}".format(best_res_val))
        print("best_epoch: {0}".format(str(best_epoch)))

        if 0 < cfg['TRAIN_TILL_EPOCH'] <= (epoch - best_epoch):
            break

    # print("Loading Best Model ...")
    model = torch.load(cfg['MODEL_SAVE_FILE'])
    return model, best_epoch


def split_list(list1d, per):
    assert sum(per) == 100
    prv = 0
    size = len(list1d)
    res = ()
    cum_per = 0
    split_count = []
    for p in per:
        cum_per += p
        nxt = int((cum_per / 100) * size)
        res = res + (itertools.islice(list1d, prv, nxt),)
        split_count.append(nxt - prv)
        prv = nxt

    return res, split_count


def single_run(cfg):
    MAX_IMG_SIZE = cfg['MAX_IMG_SIZE']
    dataset_manager = HistDataset(cfg['CLASS_TYPE'], get_path(cfg['IMAGE_FOLDER']),
                                  get_path(cfg['LABEL_FOLDER']), cfg['SPLIT'],
                                  label2id=cfg['LABEL_2_ID'], randomize=cfg['INIT_RANDOMIZE'])

    print("Prepping Dataset ...")

    test_loader = DataLoader(dataset_manager.test, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['NUM_WORKERS'],
                             collate_fn=batchify)

    print("Loading Dataset ...")

    print("Training ...")
    the_model, best_en = build_model(dataset_manager, cfg)

    print("Testing ...")
    test_eval, pred_list, true_list = test_a_epoch("test", test_loader, the_model,
                                                   get_path(cfg['TEST_RESULT_FILE']), cfg,
                                                   en=best_en)

    print("Writing Brat File ...")
    test_eval.print_results()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-y', '--yaml', dest="yaml", help="Yaml config file")

    args = parser.parse_args()
    with open(args.yaml, 'r') as cfgfile:
        config = yaml.load(cfgfile)

    single_run(config)
