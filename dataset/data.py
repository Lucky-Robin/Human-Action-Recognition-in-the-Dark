from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms



class ARID (Dataset):
    def __init__(self, root_dir, label_dir, mapping_label):
        self.root_dir = root_dir    # preprocessing/train
        self.label_dir = label_dir  # Drink
        self.path = os.path.join(self.root_dir, self.label_dir)  # preprocessing/train/Drink
        self.img_path = os.listdir(self.path)   # Drink_3_1_0.jpg
        self.mapping_label = mapping_label

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        tensor_trans = transforms.ToTensor()
        tensor_img = tensor_trans(img)
        label_alpha = self.label_dir    # Drink
        label = self.mapping_label.index(label_alpha)

        return tensor_img, label, img_name

    def __len__(self):
        return len(self.img_path)


class ARID_Val (Dataset):
    def __init__(self, root_dir, validation_label):
        self.root_dir = root_dir    # preprocessing/validate
        self.img_path = os.listdir(self.root_dir)   # 0_sampled_0.jpg
        self.validation_label = validation_label

    def __getitem__(self, idx):
        img_name = self.img_path[idx]   # 0_sampled_0.jpg
        img_item_path = os.path.join(self.root_dir, img_name)   # preprocessing/validate/0_sampled_0.jpg
        img = Image.open(img_item_path)
        tensor_trans = transforms.ToTensor()
        tensor_img = tensor_trans(img)
        video_name = img_name.split('_', 1)[0]
        video_index = int(video_name)
        labels = self.validation_label[video_index]
        label = labels
        return tensor_img, label, img_name, video_name

    def __len__(self):
        return len(self.img_path)


