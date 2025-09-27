import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .mot20 import build as build_mot20
from .mot17 import build as build_mot17
#SHA 没用到，这里展示出来就是展示一下data_path的作用
data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
    'mot20': "/home/data_SSD/zk/MOT20",
    'mot17': "/home/data_SSD/zk/MOT17"

}
#这里允许多个数据集的导入，方便变换
def build_dataset(image_set, args):
    if args.dataset_file == 'MOT17':
        return build_mot17(image_set, args)
    elif args.dataset_file == 'MOT20':
        return build_mot20(image_set, args)
    elif args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')
