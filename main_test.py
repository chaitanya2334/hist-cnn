import itertools
from argparse import ArgumentParser

import torch
import yaml
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import HistDataset
from postprocessing.evaluator import Evaluator
from utils import fit_image, argmax, get_path

MAX_IMG_SIZE = (512, 512)


def batchify(samples):
    img_size = MAX_IMG_SIZE
    images = [fit_image(sample.image, img_size) for sample in samples if sample]
    images = np.stack(images, axis=0)
    labels = [sample.label for sample in samples if sample]
    return images, labels


def test_a_epoch(name, data, model, result_file, cfg, en):
    pred_list = []
    true_list = []
    evaluator = Evaluator(name, label2id=cfg['LABEL_2_ID'], pure_labels=cfg['PURE_LABELS'])

    for images, labels in tqdm(data, desc=name, total=np.math.ceil(len(data))):
        seq_out = model(Variable(torch.from_numpy(images).float().cuda()))

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


def test_run(cfg):
    dataset = HistDataset(cfg['CLASS_TYPE'], get_path(cfg['TEST_FOLDER']),
                          get_path(cfg['LABEL_FOLDER']), split=[0, 0, 100],
                          label2id=cfg['LABEL_2_ID'], randomize=cfg['INIT_RANDOMIZE'])

    print("Prepping Dataset ...")
    test_loader = DataLoader(dataset.test, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['NUM_WORKERS'],
                             collate_fn=batchify)

    model = torch.load(cfg['MODEL_SAVE_FILE'])

    test_a_epoch("ultimate_test", test_loader, model, cfg['ULTI_RESULT_FILE'], cfg, 0)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-y', '--yaml', dest="yaml", help="Yaml config file")

    args = parser.parse_args()
    with open(args.yaml, 'r') as cfgfile:
        config = yaml.load(cfgfile)

    test_run(config)
