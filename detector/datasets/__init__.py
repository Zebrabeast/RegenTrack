import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .DroneBird import build as build_dronebird
from .DRONE import build as build_dronecrowd

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
    'DroneBird':'./data/DroneBird', 
    'Dronecrowd':'./data/Dronecrowd',      
}
# here allow other datasets ,easy to transform 
def build_dataset(image_set, args):
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
     elif args.dataset_file == 'Dronecrowd':
        return build_dronecrowd(image_set,args)
    elif args.dataset_file == 'DroneBird':
        return build_dronebird(image_set,args)
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')
