from unittest import TestCase, main

import itertools
from torch.utils.data import DataLoader

from dataset import HistDataset
import config as cfg


class TestHistDataset(TestCase):
    def test_collection(self):
        d = HistDataset().collection(cfg.IMAGE_FOLDER, cfg.LABEL_FOLDER)
        self.__test_folder(d)
        for s in d:
            self.__test_image_size(s)

    def __test_image_size(self, s):
        image = s.image
        self.assertLessEqual((image.shape[0], image.shape[1]), cfg.MAX_IMG_SIZE)

    def __test_folder(self, d):
        self.assertIsNotNone(next(d), "Either the folder is empty or the dataset is not populating any samples")

    def test_label_conversion(self):
        d = HistDataset(cfg.IMAGE_FOLDER, cfg.LABEL_FOLDER)
        loader = DataLoader(d, batch_size=1, num_workers=4, collate_fn=lambda x: x[0])

    def test_poor_read(self):
        d = HistDataset(cfg.IMAGE_FOLDER, cfg.LABEL_FOLDER, cfg.LABEL_2_ID)
        loader = DataLoader(d, batch_size=1, num_workers=4, collate_fn=lambda x: x[0])

        sli = itertools.islice(loader, 11000, 12000)
        for sample in sli:
            img = sample.image


    def test_always_fail(self):
        self.fail()

