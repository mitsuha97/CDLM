import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric

# save_UCM  save_NWPU45 save_WHURS19
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=400)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save_NWPU45_resnet18/proto_max_dis-5-5-1')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=4, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=4, pin_memory=True)

    model = Convnet().cuda()
#    model.load_state_dict(torch.load('./save/proto-1-----new7-1/max-acc.pth'))   
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    loss_fn1 = torch.nn.MSELoss(reduce='true',size_average=True)
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        

        model.train()

        tl = Averager()
        ta = Averager()


        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)

            support = proto


            test = proto.reshape(args.shot, args.train_way, -1)

            ###
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            ppp = proto
            support = support.reshape(args.shot, args.train_way, -1)
            ################################################

            support1_set, support2_set,support3_set,support4_set,support5_set = support[:,0,:],support[:,1,:],support[:,2,:],support[:,3,:],support[:,4,:]
            new_support = torch.cat((support1_set,support2_set,support3_set,support4_set,support5_set))
#            print(new_support.shape)
            new_support = new_support.view(25,64,-1)

            ##改为原型特征图
            proto = proto.view(5, 64, -1)
            old_proto = proto
            

###########
            Different_proto_for_1 = new_support[5:, : , :].mean(0).repeat(5,1,1)
            Different_proto_for_2 = torch.cat((new_support[:5, : , :], new_support[10:, : , :]), dim = 0).mean(0).repeat(5,1,1)
            Different_proto_for_3 = torch.cat((new_support[:10, : , :], new_support[15:, : , :]), dim = 0).mean(0).repeat(5,1,1)
            Different_proto_for_4 = torch.cat((new_support[:15, : , :], new_support[20:, : , :]), dim = 0).mean(0).repeat(5,1,1)
            Different_proto_for_5 = new_support[:20, : , :].mean(0).repeat(5,1,1)
#### 每个类别距离别的均值最大的均值的查询集索引
            _, Dis_proto1_index = ((new_support[:5, : , :] - Different_proto_for_1) ** 2).view(5, -1).sum(dim = 1).max(dim = 0)
            _, Dis_proto2_index = ((new_support[5:10, : , :] - Different_proto_for_2) ** 2).view(5, -1).sum(dim = 1).max(dim = 0)
            _, Dis_proto3_index = ((new_support[10:15, : , :] - Different_proto_for_3) ** 2).view(5, -1).sum(dim = 1).max(dim = 0)
            _, Dis_proto4_index = ((new_support[15:20, : , :] - Different_proto_for_4) ** 2).view(5, -1).sum(dim = 1).max(dim = 0)
            _, Dis_proto5_index = ((new_support[20:25, : , :] - Different_proto_for_5) ** 2).view(5, -1).sum(dim = 1).max(dim = 0)

            class1_support = new_support[ :5, : , :]
            class2_support = new_support[5:10, : , :]
            class3_support = new_support[10:15, : , :]
            class4_support = new_support[15:20, : , :]
            class5_support = new_support[20:25, : , :]

            class_max_dis = torch.cat((class1_support[Dis_proto1_index.item(), :, :], class2_support[Dis_proto2_index.item(), :, :], class3_support[Dis_proto3_index.item(), :, :], class4_support[Dis_proto4_index.item(), :, :], class5_support[Dis_proto5_index.item(), :, :]), dim = 0)
            loss3 = loss_fn1(class_max_dis.view(5, -1), old_proto.view(5, -1))

###########


            ################################################

            label = torch.arange(args.train_way).repeat(args.query)
            ###
#            print(label)
            ###
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), ppp)
            loss1 = F.cross_entropy(logits, label)

            loss = loss1 + loss3
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss1={:.6f} loss3={:.6f} acc={:.6f}'
                  .format(epoch, i, len(train_loader), loss1.item(), loss3.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss1 = None; loss2 = None ;new_proto =None; old_proto = None

        lr_scheduler.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
#            loss2 = loss_fn1(new_proto, old_proto)
#            loss = loss1 + loss2 
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            
            proto = None; logits = None; loss1 = None; loss2 = None ;new_proto =None; old_proto = None
        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.6f} acc={:.6f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

