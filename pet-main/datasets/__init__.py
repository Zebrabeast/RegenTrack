import torch.utils.data
import torchvision

from .SHA import build as build_sha
#SHA 没用到，这里展示出来就是展示一下data_path的作用
data_path = {
    'SHA': './data/ShanghaiTech/part_A/'    
}
# here allow other datasets ,easy to transform 
def build_dataset(image_set, args):
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')
