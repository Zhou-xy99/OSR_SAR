import os
import time
import random
import skimage.io
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision as tv
from torchvision import transforms, datasets


class self_Dataset(Dataset):
    def __init__(self, data, label=None):
        super(self_Dataset, self).__init__()

        self.data = data
        self.label = label
    def __getitem__(self, index):
        data = self.data[index]
        # data = np.moveaxis(data, 3, 1)
        # data = data.astype(np.float32)

        if self.label is not None:
            label = self.label[index]
            return data, label
        else:
            return data, 1
    def __len__(self):
        return len(self.data)

def count_data(data_dict):
    num = 0
    for key in data_dict.keys():
        num += len(data_dict[key])
    return num

class self_DataLoader(Dataset):
    def __init__(self, root, train=True, dataset='MSTAR', seed=1, nway=5):
        super(self_DataLoader, self).__init__()

        self.seed = seed
        self.nway = nway
        self.num_labels = 100
        self.SAR_num_labels = 10
        self.input_channels = 3
        self.size = 32

        self.transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5071, 0.4866, 0.4409],
                [0.2673, 0.2564, 0.2762])
            ])
        self.transform_SAR = tv.transforms.Compose([
            tv.transforms.Grayscale(1),  # 单通道
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])

        self.full_train_dict, self.full_test_dict, self.few_data_dict = self.load_data(root, dataset)

        print('full_train_num: %d' % count_data(self.full_train_dict))
        print('full_test_num: %d' % count_data(self.full_test_dict))
        print('few_data_num: %d' % count_data(self.few_data_dict))

    def load_data(self, root, dataset):
        if dataset == 'MSTAR':
            # few_selected_label = random.Random(self.seed).sample(range(self.SAR_num_labels), 1)
            few_selected_label = [4]
            print('selected labeled', few_selected_label)

            full_data_dict = {}
            few_data_dict = {}

            train_dataset = datasets.ImageFolder(root=os.path.join(root, "MSTAR"), transform=self.transform_SAR)

            train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
            for i, (data, label) in enumerate(train_loader):
                label = label.item()
                data = data.squeeze(0)
                if label in few_selected_label:
                    data_dict = few_data_dict
                else:
                    data_dict = full_data_dict

                if label not in data_dict:
                    data_dict[label] = [data]
                else:
                    data_dict[label].append(data)
            x = list(full_data_dict.values())
            x_ = list(full_data_dict.values())
            y = list(full_data_dict.keys())

            for i in range(9):
                n=int(len(x[i])*0.8)
                x[i] = x[i][n:]
                x_[i] = x_[i][:n]

            full_train_dict = dict(zip(y, x_))
            full_test_dict = dict(zip(y, x))

        else:
            raise NotImplementedError


        return full_train_dict, full_test_dict, few_data_dict


    def load_batch_data(self, train=True, batch_size=16, nway=5, num_shots=1):
        if train:
            data_dict = self.full_train_dict
            x = []
            label_y = []  # fake label: from 0 to (nway - 1)
            one_hot_y = []  # one hot for fake label
            class_y = []  # real label

            xi = []
            label_yi = []
            one_hot_yi = []

            map_label2class = []

            ### the format of x, label_y, one_hot_y, class_y is
            ### [tensor, tensor, ..., tensor] len(label_y) = batch size
            ### the first dimension of tensor = num_shots

            for i in range(batch_size):

                # sample the class to train
                sampled_classes = random.sample(data_dict.keys(), nway + 1)

                if i < (batch_size/2):
                    positive_class = random.randint(0, nway - 1)
                else:
                    positive_class = nway

                label2class = torch.LongTensor(nway + 1)

                single_xi = []
                single_one_hot_yi = []
                single_label_yi = []
                single_class_yi = []

                for j, _class in enumerate(sampled_classes):
                    if positive_class == nway:
                        if j == positive_class:
                            ### without loss of generality, we assume the 0th
                            ### sampled  class is the target class
                            sampled_data = random.sample(data_dict[_class], 1)
                            x.append(sampled_data[0])
                            label_y.append(torch.LongTensor([j]))

                            one_hot = torch.zeros(nway + 1)
                            one_hot[j] = 1.0
                            one_hot_y.append(one_hot)

                            class_y.append(torch.LongTensor([_class]))

                            shots_data = sampled_data[1:]
                        else:
                            shots_data = random.sample(data_dict[_class], num_shots)
                    else:
                        if j == positive_class and j != nway:
                            ### without loss of generality, we assume the 0th
                            ### sampled  class is the target class
                            sampled_data = random.sample(data_dict[_class], num_shots + 1)
                            x.append(sampled_data[0])
                            label_y.append(torch.LongTensor([j]))

                            one_hot = torch.zeros(nway + 1)
                            one_hot[j] = 1.0
                            one_hot_y.append(one_hot)

                            class_y.append(torch.LongTensor([_class]))

                            shots_data = sampled_data[1:]
                        elif j != positive_class and j != nway:
                            shots_data = random.sample(data_dict[_class], num_shots)
                        else:
                            continue

                    single_xi += shots_data
                    single_label_yi.append(torch.LongTensor([j]).repeat(num_shots))
                    one_hot = torch.zeros(nway + 1)
                    one_hot[j] = 1.0
                    single_one_hot_yi.append(one_hot.repeat(num_shots, 1))

                    label2class[j] = _class

                shuffle_index = torch.randperm(num_shots * nway)
                xi.append(torch.stack(single_xi, dim=0)[shuffle_index])
                label_yi.append(torch.cat(single_label_yi, dim=0)[shuffle_index])
                one_hot_yi.append(torch.cat(single_one_hot_yi, dim=0)[shuffle_index])

                map_label2class.append(label2class)

            # #x 指 test set 数据，label_y 是它对应的 label（注意，并不是在原数据集中的分类，而是在本次 task 中的分类序号），
            # #one_hot_y 是 label_y 对应的 one_hot 编码，class_y 是它对应于原数据集的分类；
            # #xi 值 support set 数据，其它带 i 的数据项，也都是指 support set 的对应部分，其解释与不带 i 的相同。
            return [torch.stack(x, 0), torch.cat(label_y, 0), torch.stack(one_hot_y, 0), \
                    torch.cat(class_y, 0), torch.stack(xi, 0), torch.stack(label_yi, 0), \
                    torch.stack(one_hot_yi, 0), torch.stack(map_label2class, 0)]
        else:
            data_dict = self.full_test_dict
            test_data_dict = self.few_data_dict
            x = []
            label_y = []  # fake label: from 0 to (nway - 1)
            one_hot_y = []  # one hot for fake label
            class_y = []  # real label

            xi = []
            label_yi = []
            one_hot_yi = []

            map_label2class = []

            ### the format of x, label_y, one_hot_y, class_y is
            ### [tensor, tensor, ..., tensor] len(label_y) = batch size
            ### the first dimension of tensor = num_shots

            for i in range(batch_size):

                # sample the class to train
                sampled_classes = random.sample(data_dict.keys(), nway)
                # print(sampled_classes)

                test_sampled_classes = random.sample(test_data_dict.keys(), 1)

                positive_class = random.randint(0, nway-1)
                # positive_class = random.randint(0, nway)
                # positive_class = nway

                label2class = torch.LongTensor(nway + 1)

                single_xi = []
                single_one_hot_yi = []
                single_label_yi = []

                if positive_class == nway:
                    for j, _class in enumerate(sampled_classes):
                        shots_data = random.sample(data_dict[_class], num_shots)

                        single_xi += shots_data
                        single_label_yi.append(torch.LongTensor([j]).repeat(num_shots))
                        one_hot = torch.zeros(nway + 1)
                        one_hot[j] = 1.0
                        single_one_hot_yi.append(one_hot.repeat(num_shots, 1))

                        label2class[j] = _class

                    sampled_data = random.sample(test_data_dict[test_sampled_classes[0]], 1)
                    x.append(sampled_data[0])
                    label_y.append(torch.LongTensor([nway]))

                    one_hot = torch.zeros(nway + 1)
                    one_hot[nway] = 1.0
                    one_hot_y.append(one_hot)

                    class_y.append(torch.LongTensor([10]))

                    shots_data = sampled_data[1:]

                    single_xi += shots_data
                    single_label_yi.append(torch.LongTensor([nway]).repeat(num_shots))

                    single_one_hot_yi.append(one_hot.repeat(num_shots, 1))

                    label2class[nway] = _class



                else:
                    for j, _class in enumerate(sampled_classes):
                        if j == positive_class:
                            ### without loss of generality, we assume the 0th
                            ### sampled  class is the target class
                            sampled_data = random.sample(data_dict[_class], num_shots + 1)
                            x.append(sampled_data[0])
                            label_y.append(torch.LongTensor([j]))

                            one_hot = torch.zeros(nway + 1)
                            one_hot[j] = 1.0
                            one_hot_y.append(one_hot)

                            class_y.append(torch.LongTensor([_class]))

                            shots_data = sampled_data[1:]
                        else:
                            shots_data = random.sample(data_dict[_class], num_shots)

                        single_xi += shots_data
                        single_label_yi.append(torch.LongTensor([j]).repeat(num_shots))
                        one_hot = torch.zeros(nway + 1)
                        one_hot[j] = 1.0
                        single_one_hot_yi.append(one_hot.repeat(num_shots, 1))

                        label2class[j] = _class

                shuffle_index = torch.randperm(num_shots * nway)
                xi.append(torch.stack(single_xi, dim=0)[shuffle_index])
                label_yi.append(torch.cat(single_label_yi, dim=0)[shuffle_index])
                one_hot_yi.append(torch.cat(single_one_hot_yi, dim=0)[shuffle_index])

                map_label2class.append(label2class)

            # #x 指 test set 数据，label_y 是它对应的 label（注意，并不是在原数据集中的分类，而是在本次 task 中的分类序号），
            # #one_hot_y 是 label_y 对应的 one_hot 编码，class_y 是它对应于原数据集的分类；
            # #xi 值 support set 数据，其它带 i 的数据项，也都是指 support set 的对应部分，其解释与不带 i 的相同。
            return [torch.stack(x, 0), torch.cat(label_y, 0), torch.stack(one_hot_y, 0), \
                    torch.cat(class_y, 0), torch.stack(xi, 0), torch.stack(label_yi, 0), \
                    torch.stack(one_hot_yi, 0), torch.stack(map_label2class, 0)]




    def load_tr_batch(self, batch_size=16, nway=5, num_shots=1):
        return self.load_batch_data(True, batch_size, nway, num_shots)

    def load_te_batch(self, batch_size=16, nway=1, num_shots=1):
        return self.load_batch_data(False, batch_size, nway, num_shots)

    def get_data_list(self, data_dict):
        data_list = []
        label_list = []
        for i in data_dict.keys():
            for data in data_dict[i]:
                data_list.append(data)
                label_list.append(i)

        now_time = time.time()

        random.Random(now_time).shuffle(data_list)
        random.Random(now_time).shuffle(label_list)

        return data_list, label_list


if __name__ == '__main__':
    D = self_DataLoader('data', True)

    # [x, label_y, one_hot_y, class_y, xi, label_yi, one_hot_yi, class_yi] = \
    #     D.load_tr_batch(batch_size=16, nway=3, num_shots=10)

    [x, label_y, one_hot_y, class_y, xi, label_yi, one_hot_yi, class_yi] = \
        D.load_te_batch(batch_size=16, nway=5, num_shots=5)

    print(x.size(), label_y.size(), one_hot_y.size(), class_y.size())
    print(xi.size(), label_yi.size(), one_hot_yi.size(), class_yi.size())
    print(label_y)
    # print(one_hot_y)
    print(class_y)
    print(label_yi)
    print(class_yi)



