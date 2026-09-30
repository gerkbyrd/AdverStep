from enum import Enum, auto
from pathlib import Path
import os
import csv
import random
from tqdm import tqdm
import copy

import torch
from datasets import load_dataset
import torchvision.transforms.v2 as T
from PIL import Image

class DatasetSource(Enum):
    IMAGENET_FATIMA = auto()
    NIPS_17 = auto()
    MSCOCO = auto()
    FLICKR = auto()
    SUPERSET = auto()

class Dataset:
    def __init__(self, source: DatasetSource, device: str = "mps", data: tuple = None, root_source: DatasetSource = None):
        self.idx2label = self.read_idx2label()
        self.source = source
        self.root_source = root_source if root_source else self.source
        self.device = device
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((224,224)),
        ])


        if source==DatasetSource.IMAGENET_FATIMA:
            self.load_fatima()
        elif source==DatasetSource.NIPS_17:
            self.load_nips17()
        elif source==DatasetSource.MSCOCO:
            self.load_mscoco()
        elif source==DatasetSource.FLICKR:
            self.load_flickr()
        elif source== DatasetSource.SUPERSET:
            self.load_from_data(data)

        #IS THIS NEEDED???
        """
        if source not in [DatasetSource.SUPERSET, DatasetSource.NIPS_17]:
            self.targets = self.compute_attack_targets()

        self.labels_str = [self.idx2label_fn(l) for l in self.labels]
        self.targets_str = [self.idx2label_fn(t) for t in self.targets]
        if self.root_source not in [DatasetSource.MSCOCO, DatasetSource.FLICKR]:
            self.labels_str = [self.idx2label_fn(l) for l in self.labels]
            self.targets_str = [self.idx2label_fn(t) for t in self.targets]
        else:
            # there are not class id labels for captioning datasets
            self.labels_str = copy.deepcopy(self.labels)
            self.targets_str = copy.deepcopy(self.targets)
        """

    def load_fatima(self,):
        self.name = "Multimodal-Fatima/Imagenet1k_sample_validation"
        dataset_hf = load_dataset(self.name, split='validation')
        dataset_hf = dataset_hf[:]

        self.images = dataset_hf['image']
        self.labels = dataset_hf['label']

    def load_flickr(self,):
        self.name = "FLICKR"
        here = Path(os.path.abspath(__file__)).parent
        images_path = (here /".." /  "archive_flickr" / "images").resolve()
        label_path = (here /".." /  "archive_flickr" / "images.csv").resolve()

        self.name_labels, self.name_targets = {}, {}
        with open(label_path) as f:
            reader = csv.reader(f)
            for line in list(reader)[1:]:
                name, label = line[3], line[-2].split('"')[1]
                self.name_labels[name] = self.name_targets[name] = label

        self.image_files = os.listdir(images_path)
        self.image_files.sort()

        self.images, self.labels, self.targets = [], [], []
        for i in range(len(self.image_files)):
            name = self.image_files[i]
            tname = self.image_files[len(self.image_files) - 1 - i]
            self.images.append(Image.open(os.path.join(images_path, name)))
            self.labels.append(self.name_labels[name])
            self.targets.append(self.name_targets[tname])


    def load_mscoco(self,):
        self.name = "MSCOCO"
        here = Path(os.path.abspath(__file__)).parent
        images_path = (here /".." /  "archive_coco" / "images").resolve()
        label_path = (here /".." /  "archive_coco" / "images.csv").resolve()

        self.name_labels, self.name_targets = {}, {}
        with open(label_path) as f:
            reader = csv.reader(f)
            for line in list(reader)[1:]:
                name, label = line[2], line[-2].split('"')[1]
                #print(label)
                self.name_labels[name] = self.name_targets[name] = label
                #input(self.name_labels[name])
        self.image_files = os.listdir(images_path)
        self.image_files.sort()

        self.images, self.labels, self.targets = [], [], []
        for i in range(len(self.image_files)):
            name = self.image_files[i]
            tname = self.image_files[len(self.image_files) - 1 - i]
            self.images.append(Image.open(os.path.join(images_path, name)))
            self.labels.append(self.name_labels[name])
            self.targets.append(self.name_targets[tname])

    def load_nips17(self,):
        self.name = "NIPS17"
        here = Path(os.path.abspath(__file__)).parent
        images_path = (here /".." /  "archive" / "images").resolve()
        label_path = (here /".." /  "archive" / "images.csv").resolve()

        self.name_labels, self.name_targets = {}, {}
        with open(label_path) as f:
            reader = csv.reader(f)
            for line in list(reader)[1:]:
                name, label, target = line[0], int(line[6]) - 1, int(line[7]) - 1
                self.name_labels[name + '.png'] = label
                self.name_targets[name + '.png'] = target

        self.image_files = os.listdir(images_path)
        self.image_files.sort()

        self.images, self.labels, self.targets = [], [], []
        for i in range(len(self.image_files)):
            name = self.image_files[i]
            self.images.append(Image.open(os.path.join(images_path, name)))
            self.labels.append(self.name_labels[name])
            self.targets.append(self.name_targets[name])

    def load_from_data(self, data):
        self.images, self.labels, self.targets = data

    def __getitem__(self, i):
        assert type(i) == int, f"Cannot handle selecting from the dataset with type {type(i)}"

        return self.transform(self.images[i]), self.labels[i], self.targets[i]

    def __len__(self,):
        return len(self.images)

    def get_unique_labels(self,):
        return list(set(self.labels))

    def compute_attack_targets(self,) -> list:
        if self.source not in [DatasetSource.MSCOCO, DatasetSource.FLICKR]:
            # pairwise cosine similarities of all labels
            cs_matrix = self.read_label_similarities()

            unique_labels = torch.tensor(self.get_unique_labels()).to(self.device)
            targets = []
            for l in tqdm(self.labels, desc="Computing attack targets"):
                valid_targets = torch.argwhere(cs_matrix[l,:] < torch.median(cs_matrix[l,:]))
                valid_targets = valid_targets[torch.isin(valid_targets, unique_labels)].tolist()
                targets.append(random.choice(valid_targets))
        else:
            # shuffle labels, but ensure that no labels stays the same by chance
            targets = self.labels[:]

            # all image captions have to be shuffled
            while any(targets[i] == self.labels[i] for i in range(len(self.labels))):
                random.shuffle(targets)

        return targets


    def split_train_test(self, train_ratio: float) -> tuple:
        limit = int(len(self)*train_ratio)
        train_ds = Dataset(source=DatasetSource.SUPERSET, data=(self.images[:limit], self.labels[:limit], self.targets[:limit]), root_source=self.root_source)
        test_ds = Dataset(source=DatasetSource.SUPERSET, data=(self.images[limit:], self.labels[limit:], self.targets[limit:]), root_source=self.root_source)
        return train_ds, test_ds

    def create_susbset(self, subset_len: int, is_random: bool = False):
        if is_random:
            chosen_ids = random.choices(range(len(self)), k=subset_len)
        else:
            chosen_ids = [i for i in range(subset_len)]

        images = self.select(self.images, chosen_ids)
        labels = self.select(self.labels, chosen_ids)
        targets = self.select(self.targets, chosen_ids)

        return Dataset(source=DatasetSource.SUPERSET, data=(images, labels, targets), root_source=self.root_source)

    @staticmethod
    def select(population: list, indices: list[int]):
        return [population[i] for i in indices]

    @staticmethod
    def read_idx2label():
        cls_file = Path(os.path.abspath(__file__)).parent.parent.parent / "data" / "imagenet1000_clsidx_to_labels.txt"
        with open(cls_file, "r") as f:
            idx2label = eval(f.read())
        return idx2label

    @staticmethod
    def read_label_similarities():
        sim_file = Path(os.path.abspath(__file__)).parent.parent.parent / "data" / "imagenet_clip_similarities.pt"
        return torch.load(sim_file)

    def idx2label_fn(self, idx) -> str:
        return self.idx2label[idx].split(",")[0].strip()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # important for choosing the same target labels each time
    random.seed(42)

    ds = Dataset(DatasetSource.NIPS_17)


    # create train and test splits
    ds_train, ds_test = ds.split_train_test(train_ratio=0.8)
    print(len(ds_train), len(ds_test))

    # create random subset and loop over it
    ds_subset = ds.create_susbset(subset_len=10, is_random=False)
    dataloader = DataLoader(ds_subset, batch_size=4, shuffle=False)

    for img, lbl, tgt in dataloader:
        print(img.shape, lbl, tgt)
