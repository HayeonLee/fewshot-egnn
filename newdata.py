import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import numpy as np
import torch


def randsamp(data, args, shuf_cls=True, is_train=True):
    w = args.way; s = args.shot; q = args.query
    ow = args.oway; os = args.oshot; oq = args.oquery

    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if shuf_cls:
            rand = torch.randperm(len(data)).to(device)  
            if args.use_test_inner_loop or is_train:                                                                                  
                for i in range(w):
                    cls = data[rand[i]].to(device)
                    _shot, _query = sample_instances(cls, args, s, q)
                    if i == 0:
                        shot = _shot # [1, 5, 3, 84, 84]
                        query = _query # [1, 15, 3, 84, 84]
                    else:
                        shot = torch.cat((shot, _shot))
                        query = torch.cat((query, _query))
                shot = to_shot_cls(shot, args.xdim, args.size)
                query = to_shot_cls(query, args.xdim, args.size)
            else:
                shot = None; query = None

        if ow is not None:
            for i in range(w, w + ow):
                cls = data[rand[i]].to(device)
                _shot, _query = sample_instances(cls, args, os, oq)
                if i == w:
                    oshot = _shot
                    oquery = _query
                else:
                    oshot = torch.cat((oshot, _shot))
                    oquery = torch.cat((oquery, _query))
            oshot = to_shot_cls(oshot, args.xdim, args.size)
            oquery = to_shot_cls(oquery, args.xdim, args.size)
        else:
            oshot = None; oquery = None
    return shot, query, oshot, oquery


def TieredImageNet(setname='train', ROOT_PATH='../../materials'):
    st = time.time()
    if setname == 'train':
        path = osp.join(ROOT_PATH,'tiered-imagenet',  'imtiered_' + setname + '-1.pt')
        data = pickle.load(open(path, 'rb'))
        path = osp.join(ROOT_PATH, 'tiered-imagenet', 'imtiered_' +  setname + '-2.pt')
        data += pickle.load(open(path, 'rb'))
    else:
        path = osp.join(ROOT_PATH, 'tiered-imagenet', 'imtiered_' +  setname + '.pt')
        print(path)
        data = pickle.load(open(path, 'rb'))
    dt = time.time() - st
    #datas = []
    #for i in range(9):
    #    for j in range(11):
    #        datas.append(data[j])
    #with open('/st1/hayeon/materials/tiered-imagenet/imtiered_debug.pt', 'wb') as f:
    #    pickle.dump(datas, f)

    print('=> [{:.3f}min] {} data size: {}'.format(dt/60, setname, len(data)))
    return data



#ROOT_PATH = '../materials/'
def get_dataset(data_name='mini', root_path='../materials', mode_list=['train', 'test']):
    dataset = {}
    for mode in mode_list:
        if data_name == 'omni':
            dataset[mode] = Omniglot(mode, root_path)
        elif data_name == 'omni_rot':
            dataset[mode] = OmniglotRot(mode, root_path)
        elif data_name == 'mini':
            dataset[mode] = MiniImageNet(mode, root_path) # [64, 600, 3, 84, 84]
        elif data_name == 'tiered':
            dataset[mode] = TieredImageNet(mode, root_path) # [64, 600, 3, 84, 84]

    return dataset


# tr: [64, 600, 84, 84, 3]
# val: [16, 600, 84, 84, 3]
# test: [20, 600, 84, 84, 3]
def MiniImageNet(setname='train', ROOT_PATH='../materials'):
    path = osp.join(ROOT_PATH, 'mini-imagenet', setname + '.npy')
    data = np.load(path)
    # data = data.reshape(-1, 600, 84, 84, 3)
    # data = data.transpose(0, 1, 4, 2, 3)
    # print('=> {} data size: {}'.format(setname, np.shape(data)))
    data = torch.from_numpy(data).float() # [38400, 84, 84, 3]
    data = data.view(-1, 600, 84, 84, 3).contiguous().permute(0, 1, 4, 2, 3).cuda() # [64, 600, 3, 84, 84]
    print('=> {} data size: {}'.format(setname, data.size()))
    return data

