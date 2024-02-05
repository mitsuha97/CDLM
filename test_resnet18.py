import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric

# save_UCM  save_NWPU45_resnet18 save_WHURS19 save save_

if __name__ == '__main__':
    feature_extract_net = torchvision.models.resnet18().cuda()
    feature_extract_net.fc = nn.Sequential()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./save_UCM_resnet18/proto_max_dis-5-5-1/epoch-160.pth')
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=20)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = MiniImageNet('test')
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=2, pin_memory=True)

    feature_extract_net.load_state_dict(torch.load(args.load))
    feature_extract_net.eval()

    ave_acc = Averager()

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]

        x = feature_extract_net(data_shot).view(data_shot.size(0), -1)
        x = x.reshape(args.shot, args.way, -1).mean(dim=0)
        p = x

        logits = euclidean_metric(feature_extract_net(data_query).view(data_query.size(0), -1), p)

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
        x = None; p = None; logits = None

