import argparse
import os.path as osp
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric

# save_UCM_resnet18  save_NWPU45_resnet18 save_WHURS19_resnet18
if __name__ == '__main__':
    feature_extract_net = torchvision.models.resnet18(pretrained=True).cuda()
    feature_extract_net.fc = nn.Sequential()
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=400)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save_WHURS19_resnet18/proto_cali_mean_dis_max_dis_5_5_1')
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


#    model.load_state_dict(torch.load('./save/proto-1-----new7-1/max-acc.pth'))   
    optimizer = torch.optim.Adam(feature_extract_net.parameters(), lr=0.0008)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    def save_model(name):
        torch.save(feature_extract_net.state_dict(), osp.join(args.save_path, name + '.pth'))
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
        

        feature_extract_net.train()

        tl = Averager()
        ta = Averager()


        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = feature_extract_net(data_shot).view(data_shot.size(0), -1)

            support = proto




            ###
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            ppp = proto
            support = support.reshape(args.shot, args.train_way, -1)
            ################################################

            support1_set, support2_set,support3_set,support4_set,support5_set = support[:,0,:],support[:,1,:],support[:,2,:],support[:,3,:],support[:,4,:]
            new_support = torch.cat((support1_set,support2_set,support3_set,support4_set,support5_set))
#            print(new_support.shape)
            new_support = new_support.view(25,512,-1)

            ##改为原型特征图
            proto = proto.view(5, 512, -1)
            old_proto = proto
            
#############################################
            ##计算Loss2
            proto1, proto2, proto3, proto4, proto5, = proto[0,:,:].unsqueeze(0), proto[1,:,:].unsqueeze(0), proto[2,:,:].unsqueeze(0), proto[3,:,:].unsqueeze(0), proto[4,:,:].unsqueeze(0)

            p1_repeat = proto1.repeat(5,1,1)
            p2_repeat = proto2.repeat(5,1,1)
            p3_repeat = proto3.repeat(5,1,1)
            p4_repeat = proto4.repeat(5,1,1)
            p5_repeat = proto5.repeat(5,1,1)

            p1_similarity_matrix = (((p1_repeat - new_support[ :5, : , :])**2)).mean(0)
            p2_similarity_matrix = (((p2_repeat - new_support[5:10, : , :])**2)).mean(0)
            p3_similarity_matrix = (((p3_repeat - new_support[10:15, : , :])**2)).mean(0)
            p4_similarity_matrix = (((p4_repeat - new_support[15:20, : , :])**2)).mean(0)
            p5_similarity_matrix = (((p5_repeat - new_support[20:25, : , :])**2)).mean(0)


            proto1 = (p1_similarity_matrix + 1) * proto1
            proto2 = (p2_similarity_matrix + 1) * proto2
            proto3 = (p3_similarity_matrix + 1) * proto3
            proto4 = (p4_similarity_matrix + 1) * proto4
            proto5 = (p5_similarity_matrix + 1) * proto5
###########################################
            ##计算距离的方差
            p1_similarity_matrix_repeat_5 = (((p1_repeat - new_support[ :5, : , :])**2)).mean(0).repeat(5,1,1)
            p2_similarity_matrix_repeat_5 = (((p2_repeat - new_support[5:10, : , :])**2)).mean(0).repeat(5,1,1)
            p3_similarity_matrix_repeat_5 = (((p3_repeat - new_support[10:15, : , :])**2)).mean(0).repeat(5,1,1)
            p4_similarity_matrix_repeat_5 = (((p4_repeat - new_support[15:20, : , :])**2)).mean(0).repeat(5,1,1)
            p5_similarity_matrix_repeat_5 = (((p5_repeat - new_support[20:25, : , :])**2)).mean(0).repeat(5,1,1)

            p1_support_distance = (((p1_repeat - new_support[ :5, : , :])**2))
            p2_support_distance = (((p2_repeat - new_support[5:10, : , :])**2))
            p3_support_distance = (((p3_repeat - new_support[10:15, : , :])**2))
            p4_support_distance = (((p4_repeat - new_support[15:20, : , :])**2))
            p5_support_distance = (((p5_repeat - new_support[20:25, : , :])**2))

            new_proto = torch.cat((proto1, proto2, proto3, proto4, proto5))
#######################################
            ##计算最大距离
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

#############################################
            support_distance = torch.cat((p1_similarity_matrix_repeat_5, p2_similarity_matrix_repeat_5, p3_similarity_matrix_repeat_5, p4_similarity_matrix_repeat_5, p5_similarity_matrix_repeat_5),dim = 0)
            support_dis_mean = torch.cat((p1_support_distance, p2_support_distance, p3_support_distance, p4_support_distance, p5_support_distance),dim = 0)

            ################################################
            class_max_dis = torch.cat((class1_support[Dis_proto1_index.item(), :, :], class2_support[Dis_proto2_index.item(), :, :], class3_support[Dis_proto3_index.item(), :, :], class4_support[Dis_proto4_index.item(), :, :], class5_support[Dis_proto5_index.item(), :, :]), dim = 0)
            loss3 = loss_fn1(class_max_dis.view(5, -1), old_proto.view(5, -1))
            #########################3
            label = torch.arange(args.train_way).repeat(args.query)
            ###
#            print(label)
            ###
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(feature_extract_net(data_query).view(data_query.size(0), -1), ppp)
            loss1 = F.cross_entropy(logits, label)
            loss2 = loss_fn1(new_proto.view(5,-1), old_proto.view(5, -1))
            loss4 = loss_fn1(support_distance.view(25, -1), support_dis_mean.view(25, -1))
            loss = loss1 + loss2 + loss3 + loss4
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss1={:.6f} loss2={:.6f} loss3={:.6f} loss4={:.6f} acc={:.6f}'
                  .format(epoch, i, len(train_loader), loss1.item(), loss2.item(),loss3.item(), loss4.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss1 = None; loss2 = None ;loss4  = None;new_proto =None; old_proto = None

        lr_scheduler.step()

        tl = tl.item()
        ta = ta.item()

        feature_extract_net.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = feature_extract_net(data_shot).view(data_shot.size(0), -1)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(feature_extract_net(data_query).view(data_query.size(0), -1), proto)
            loss = F.cross_entropy(logits, label)
#            loss2 = loss_fn1(new_proto, old_proto)
#            loss = loss1 + loss2 
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            
            proto = None; logits = None; loss1 = None; loss2 = None ;loss4  = None;new_proto =None; old_proto = None

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

