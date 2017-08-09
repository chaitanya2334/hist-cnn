import argparse

import cv2
import numpy as np
import torch

import torchvision
import yaml

from visualize.salient_map_generators import BackProp, GradCAM, GuidedBackProp
from torchvision import transforms


def visualize(cfg):
    image = "../input/TCGA-A2-A0CM-01A-03-BSC.c240a44d-0583-496e-bdef-5cdb5e0ee167/slide/18/7_4.jpeg"
    # Load the synset words
    file_name = 'synset_words.txt'
    classes = {v: k for k, v in cfg['PURE_LABELS'].items()}

    print('Loading a model...')
    model = torch.load(cfg['MODEL_SAVE_FILE'])

    print(list(model.named_modules()))

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
    ])

    print('\nBackpropagation')
    bp = BackProp(model=model, target_layer='pre_layers.0',
                  n_class=cfg['NUM_LABELS'], cuda=True)
    bp.load_image(image, transform)
    bp.forward()

    for i in range(0, 4):
        print('{:.5f}\t{}'.format(bp.prob[i], classes[bp.idx[i]]))
        bp.backward(idx=bp.idx[i])
        bp_data = bp.generate()
        bp.save('../results/bp_{}.png'.format(classes[bp.idx[i]]), bp_data)

    print('\nGuided Backpropagation')
    gbp = GuidedBackProp(model=model, target_layer='pre_layers.0',
                         n_class=cfg['NUM_LABELS'], cuda=True)
    gbp.load_image(image, transform)
    gbp.forward()

    for i in range(0, 4):
        gbp.backward(idx=gbp.idx[i])
        gbp_data = gbp.generate()
        gbp.save(
            '../results/gbp_{}.png'.format(classes[gbp.idx[i]]), gbp_data.copy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', dest="yaml", help="Yaml config file")

    args = parser.parse_args()
    with open(args.yaml, 'r') as cfgfile:
        config = yaml.load(cfgfile)

    visualize(config)
