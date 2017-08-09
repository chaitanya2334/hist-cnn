import glob
import os
from collections import namedtuple, defaultdict
from random import shuffle

import cv2
from torch.utils import data
from tqdm import tqdm
import csv_utils

Sample = namedtuple('Sample', ['image', 'label', "wsid", "x", "y"])


class CustomDataset(data.Dataset):
    def __init__(self, metadata, label2id, labels_csv):
        self.__metadata = metadata
        self.label2id = label2id
        self.__labels_csv = labels_csv

    def __getitem__(self, item):

        filepath, label, wsid, x, y = self.__metadata[item]

        image = cv2.imread(filepath)

        if image is not None and image.data:
            return Sample(image, label, wsid, x, y)
        else:
            return Sample(None, label, wsid, x, y)

    def __len__(self):
        return len(self.__metadata)


class HistDataset(object):
    def __init__(self, class_type, image_dir, label_filepath, split, label2id=None, randomize=False):

        self.class_type = class_type

        self.label2id = label2id
        self.__labels_csv = csv_utils.read(label_filepath)

        self.__metadata_train, self.__metadata_dev, self.__metadata_test = \
            self.__read_folders(image_dir, split, randomize)

        self.train = CustomDataset(self.__metadata_train, self.label2id, self.__labels_csv)
        self.dev = CustomDataset(self.__metadata_dev, self.label2id, self.__labels_csv)
        self.test = CustomDataset(self.__metadata_test, self.label2id, self.__labels_csv)

        print(len(self.train))
        print(len(self.__metadata_train))
        print(len(self.dev))
        print(len(self.__metadata_dev))
        print(len(self.test))
        print(len(self.__metadata_test))

        assert len(self.train) + len(self.dev) + len(self.test) == \
               len(self.__metadata_train) + len(self.__metadata_dev) + len(self.__metadata_test)

    @staticmethod
    def __split_dataset(per, bins):
        res = ([], [], [])
        assert sum(per) == 100
        print(res)
        for k, l in bins.items():
            size = len(l)
            print("label: {0}, size: {1}".format(k, len(l)))
            cum_per = 0
            prv = 0
            for i, p in enumerate(per):
                cum_per += p
                nxt = int((cum_per / 100) * size)
                res[i].extend(l[prv:nxt])
                print(len(res[0]), len(res[1]), len(res[2]))
                prv = nxt

        return res

    def __label_by_id(self, wsid):

        for row in self.__labels_csv:
            if row['Complete TCGA ID'] in wsid:
                return row['Complete TCGA ID'], row[self.class_type]

        return None, None

    def __read_metadata(self, name, wspaths):
        metadata = []
        for wspath in tqdm(wspaths, desc="Reading Image Folders for {0}".format(name)):
            wsid = os.path.splitext(os.path.basename(wspath))[0]
            filepaths = glob.glob(os.path.join(wspath, "slide", "*", "*.jpeg"))
            for filepath in filepaths:
                basename = os.path.splitext(os.path.basename(filepath))[0]  # just the filename (remove extension)

                pos = basename.split("_")

                if len(pos) == 2:
                    x, y = pos
                elif len(pos) == 3:
                    x, y, a = pos
                else:
                    x = pos[0]
                    y = pos[1]

                _, label = self.__label_by_id(wsid)

                label = self.label2id[label]

                metadata.append((filepath, label, wsid, x, y))

        return metadata

    def __bin_paths(self, wspaths):
        d = defaultdict(list)
        for wspath in wspaths:
            wsid = os.path.splitext(os.path.basename(wspath))[0]
            _, label = self.__label_by_id(wsid)
            if label:
                label = self.label2id[label]
                d[label].append(wspath)

        return d

    @staticmethod
    def __print_dataset(datasets, names, labels):
        for dataset, name in zip(datasets, names):
            print(name)
            for label in labels:
                print("{0} : {1}".format(label, sum([item[1] == label for item in dataset])))

    def __read_folders(self, image_dir, split, randomize):
        # assumes the following file structure for access to images:
        # folder
        #   |
        #   ├── TCGA....
        #   |   |
        #   |   ├── label
        #   |   ├── macro
        #   |   ├── thumbnail
        #   |   ├── slide
        #   |   |   |
        #   |   |   ├── (int)
        #   |   |   |   |
        #   |   |   |   ├── <x1>_<y1>.jpeg
        #   |   |   |   ├── <x2>_<y2>.jpeg
        #   |   |   |   ├── <x3>_<y3>.jpeg
        #   ├── TCGA ....

        wspaths = sorted(glob.glob(image_dir + "/*"))

        binned_wspaths = self.__bin_paths(wspaths)

        if randomize:
            shuffle(wspaths)

        print(len(wspaths))
        wspaths_train, wspaths_dev, wspaths_test = self.__split_dataset(split, binned_wspaths)

        print(len(wspaths_train), len(wspaths_dev), len(wspaths_test))

        metadata_train = self.__read_metadata("train", wspaths_train)
        metadata_dev = self.__read_metadata("dev", wspaths_dev)
        metadata_test = self.__read_metadata("test", wspaths_test)

        self.__print_dataset((metadata_train, metadata_dev, metadata_test), ("train", "dev", "test"), binned_wspaths.keys())

        return metadata_train, metadata_dev, metadata_test

    def __all_labels(self, name):
        return [row[name] for row in self.__labels_csv]
