import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class PatientLevelRetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform
        self.test = test
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.get_item(index)

    def load_data(self):
        df = pd.read_csv(self.ann_file)

        df['prefix'] = df['image_id'].str.split('_').str[0]
        grouped = df.groupby(['prefix'])

        data = []
        for (prefix), group in grouped:
            file_info = dict()
            file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
            file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
            file_info['img_path3'] = os.path.join(self.image_dir, group.iloc[2]['img_path'])
            file_info['img_path4'] = os.path.join(self.image_dir, group.iloc[3]['img_path'])
            if not self.test:
                file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img1 = Image.open(data['img_path1']).convert('RGB')
        img2 = Image.open(data['img_path2']).convert('RGB')
        img3 = Image.open(data['img_path3']).convert('RGB')
        img4 = Image.open(data['img_path4']).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2, img3, img4], label
        else:
            return [img1, img2, img3, img4]
