import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import os
from pathlib import Path
from ruamel.yaml import YAML
import argparse
from logger import MetricLogger
from utils import get_surrogate, get_target, normalize_list
from dataset.dataCluster import DataFolderWithLabel, DataFolderWithClassNoise
from models.generator import ResnetGenerator
from models.generator3d import ResnetGenerator3d
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess
import horovod.torch as hvd
import datetime

def init_comm_size_and_rank():
    world_size = None
    world_rank = 0

    if os.getenv("OMPI_COMM_WORLD_SIZE") and os.getenv("OMPI_COMM_WORLD_RANK"):
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

    return int(world_size), int(world_rank)

def get_local_rank():
    local_rank = 0
    if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
        localrank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    return localrank

#def setup(rank, world_size):
def setup():

    if dist.is_initialized():
       world_size, world_rank = init_comm_size_and_rank()
       return world_size, world_rank

    world_size, world_rank = init_comm_size_and_rank()

    if os.getenv("LSB_HOSTS") is not None:
        master_addr = os.environ["LSB_HOSTS"].split()[1]

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["WORLD_RANK"] = str(world_rank)
    os.environ["RANK"] = str(world_rank)
    os.environ["LOCAL_RANK"] = str(get_local_rank())

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=1800)
        )
    
    return world_size, world_rank

    ## get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(os.environ['LSB_DJOB_HOSTFILE'])
    #try:
    #    result = subprocess.run("cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch", shell=True, capture_output=True, text=True, check=True)
    #    nodes = result.stdout.strip().split('\n')
    #    head = nodes[0] if nodes else None
    #    if head:
    #        os.environ['MASTER_ADDR'] = head
    #        print(f"Setting env_var MASTER_ADDR = {head}")
    #    else:
    #        print("No valid nodes found")

    #except subprocess.CalledProcessError as e:
    #    print(f"Error while getting nodes: {e}")
    ## os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
    #os.environ['MASTER_PORT'] = '29500'
    #os.environ['WORLD_SIZE'] = os.environ.get('OMPI_COMM_WORLD_SIZE', '1')
    #os.environ['RANK'] = os.environ.get('OMPI_COMM_WORLD_RANK', '0')
    #os.environ['LOCAL_RANK'] = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    #dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def get_device_name(use_gpu=True, rank_per_model=1, verbosity_level=0, no_prefix=False):

    available_gpus = get_device_list()
    if not use_gpu or not available_gpus:
        # print_distributed(verbosity_level, "Using CPU")
        return "cpu"

    world_size, world_rank = get_comm_size_and_rank()
    if rank_per_model != 1:
        raise ValueError("Exactly 1 rank per device currently supported")

    # print_distributed(verbosity_level, "Using GPU")
    ## We need to ge a local rank if there are multiple GPUs available.
    localrank = 0
    if torch.cuda.device_count() > 1:
        if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
            ## Summit
            localrank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

        if localrank >= torch.cuda.device_count():
            print(
                "WARN: localrank is greater than the available device count - %d %d"
                % (localrank, torch.cuda.device_count())
            )

    if no_prefix:
        device_name = str(localrank)
    else:
        device_name = "cuda:" + str(localrank)

    return device_name

def get_device_from_name(device_name):
    if device_name.startswith('cuda'):
        return torch.device(device_name)
    else:
        return torch.device('cpu')

def get_comm_size_and_rank():
    world_size, world_rank = init_comm_size_and_rank()
    return world_size, world_rank

def get_device_list():
    num_devices = torch.cuda.device_count()
    device_list = [torch.cuda.get_device_name(i) for i in range(num_devices)]
    return device_list

def get_distributed_model(model, verbosity=0, sync_batch_norm=False):
    device_name = get_device_name(verbosity_level=verbosity)
    print('distributed_model device name:', device_name)
    if dist.is_initialized():
        print('double check ', device_name)
        if device_name == "cpu":
            print('cpu assigned')
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            if sync_batch_norm:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            device = get_device_from_name(device_name)
            model = model.to(device)
            print('distributed model device assigned',device) 
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device]
            )
    return model, device_name

def cleanup():
    dist.destroy_process_group()

#def train_gnet(rank, world_size, args, config):
def train_gnet(args, config):
    #setup(rank, world_size)
    world_size, world_rank = setup()
    #device = torch.device(f'cuda:{rank}')
    # hvd.init()
    # torch.cuda.set_device(hvd.local_rank())
    # print(torch.cuda.get_device_name(args.device))
    train_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    #print(torch.cuda.get_device_name(args.device))
    normalize = normalize_list[config['normalize']]

    # net = get_surrogate(config['model'], config['num_classes']).eval().to(args.device)
    # sd = torch.load(config['checkpoint'], map_location='cpu')
    # net.load_state_dict(sd)

    #net = get_surrogate(config['model'], config['num_classes']).eval().to(device)
    net = get_surrogate(config['model'], config['num_classes'])
    # net = net.to(device)
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [rank], find_unused_parameters=True)
    _, device = get_distributed_model(net)
    net = net.to(device)
    # print('net device', next(net.parameters()).device)
    cluster = torch.load(config['cluster'], map_location='cpu')
    cluster['centers'] = cluster['centers'].to(device)
    num_clusters = cluster['centers'].shape[0]

    train_dataset = DataFolderWithLabel(config['dataset']['config']['train'], cluster['pred_idx'], train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, pin_memory=True)
    # train_dataset = DataFolderWithLabel(config['dataset']['config']['train'], cluster['pred_idx'], train_transform)
    #sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=world_rank)
    #train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True, sampler=sampler)
    
    # Using hvd
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replocas=hvd.size(), rank=hvd.rank())
    # train_loader = torch.utils.data.Dataloader(train_dataset, batch_size=256, num_workers=8, pin_memory=True, sampler=train_sampler)

    for cluster_idx in range(num_clusters):
        g_net = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu')
        # g_net, device = get_distributed_model(g_net)
        g_net = g_net.to(device)
        noise = torch.zeros((1, 3, 224, 224)) # 修改噪声形状
        noise.uniform_(0, 1)
        noise = noise.to(device)
        # print('g_net device:', next(g_net.parameters()).device)
        # print('noise device:', noise.device)
        # g_net = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu') # 修改生成网络通道数
        # g_net,device = get_distributed_model(g_net)
        # g_net.to(args.device)
        #g_net.cuda()

        optimizer = torch.optim.Adam(g_net.parameters(), lr=config['lr'], weight_decay=5e-4)
        # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.name_parameters())
        # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epoch'] * len(train_loader), eta_min=1e-6)
        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        logger = MetricLogger()

        features = {}
        def hook(layer, inp, out):
            features['feat'] = inp[0]
        net.fc.register_forward_hook(hook)
        # net.module.fc.register_forward_hook(hook)

        for epoch in range(config['num_epoch']):
            g_net.train()
            # sampler.set_epoch(epoch)
            header = 'Class idx {}\tTrain Epoch {}:'.format(cluster_idx, epoch)

            for images, _, _ in logger.log_every(train_loader, 50, header=header):
                images = images.to(device)
                delta_im = g_net(noise).repeat(images.shape[0], 1, 1, 1)

                if config['norm'] == 'l2':
                    temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                    delta_im = delta_im * config['epsilon'] / temp
                else:
                    delta_im = torch.clamp(delta_im, -config['epsilon'] / 255., config['epsilon'] / 255)

                images_adv = torch.clamp(images + delta_im, 0, 1)
                target_labels = (torch.ones(len(images)).long() * cluster_idx + config['target_offset']) % num_clusters
                target_labels = target_labels.to(device)
                # anchors = torch.index_select(cluster['centers'], dim=0, index=target_labels)
                anchors = torch.stack([cluster['centers'][i] for i in target_labels], dim=0).to(device)

                net(normalize(images_adv))
                loss = criterion(features['feat'].log_softmax(dim=-1), anchors.softmax(dim=-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                logger.meters['train_loss'].update(loss.item(), n=len(images))

            with torch.no_grad():
                perturbation = g_net(noise)
            torch.save({'state_dict': g_net.state_dict(), 'init_noise': noise, 'perturbation': perturbation}, os.path.join(config['output_dir'], f'perturbation_{cluster_idx}.pth'))
            logger.clear()

def train_gnet_3d(args, config):
    print(torch.cuda.get_device_name(args.device))
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    print(torch.cuda.get_device_name(args.device))
    normalize = normalize_list[config['normalize']]

    net = get_surrogate(config['model'], config['num_classes']).eval().to(args.device)
    # sd = torch.load(config['checkpoint'], map_location='cpu')
    # net.load_state_dict(sd)

    cluster = torch.load(config['cluster'], map_location='cpu')
    cluster['centers'] = cluster['centers'].to(args.device)
    num_clusters = cluster['centers'].shape[0]

    train_dataset = DataFolderWithLabel(config['dataset']['config']['train'], cluster['pred_idx'], train_transform)
    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=8, pin_memory=True)

    for cluster_idx in range(num_clusters):
        noise = torch.zeros((1, 1, 224, 224)) # 修改噪声形状
        noise.uniform_(0, 1)
        noise = noise.to(args.device)

        g_net = ResnetGenerator3d(1, 1, 64, norm_type='batch', act_type='relu') # 修改生成网络通道数
        g_net.to(args.device)

        optimizer = torch.optim.Adam(g_net.parameters(), lr=config['lr'], weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epoch'] * len(train_loader), eta_min=1e-6)
        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        logger = MetricLogger()

        features = {}
        def hook(layer, inp, out):
            features['feat'] = inp[0]
        net.fc.register_forward_hook(hook)

        for epoch in range(config['num_epoch']):
            g_net.train()
            header = 'Class idx {}\tTrain Epoch {}:'.format(cluster_idx, epoch)

            for images, _, _ in logger.log_every(train_loader, 50, header=header):
                images = images.to(args.device)
                delta_im = g_net(noise).repeat(images.shape[0], 1, 1, 1)

                if config['norm'] == 'l2':
                    temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                    delta_im = delta_im * config['epsilon'] / temp
                else:
                    delta_im = torch.clamp(delta_im, -config['epsilon'] / 255., config['epsilon'] / 255)

                images_adv = torch.clamp(images + delta_im, 0, 1)
                target_labels = (torch.ones(len(images)).long() * cluster_idx + config['target_offset']) % num_clusters
                target_labels = target_labels.to(args.device)
                # anchors = torch.index_select(cluster['centers'], dim=0, index=target_labels)
                anchors = torch.stack([cluster['centers'][i] for i in target_labels], dim=0).to(args.device)

                net(normalize(images_adv))
                loss = criterion(features['feat'].log_softmax(dim=-1), anchors.softmax(dim=-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                logger.meters['train_loss'].update(loss.item(), n=len(images))

            with torch.no_grad():
                perturbation = g_net(noise)
            torch.save({'state_dict': g_net.state_dict(), 'init_noise': noise, 'perturbation': perturbation}, os.path.join(config['output_dir'], f'perturbation_{cluster_idx}.pth'))
            logger.clear()


def train(args, config):
    train_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    normalize = normalize_list[config['normalize']]

    num_classes = config['dataset']['config']['num_classes']

    train_dataset = DataFolderWithLabel(config['ae_dir'], None, transform=train_transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'], num_workers=8, pin_memory=True)

    test_dataset = DataFolderWithLabel(config['dataset']['config']['test'], None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=8, pin_memory=True)

    net = get_target(config['model'], num_classes).to(args.device)

    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epoch'] * len(train_loader), eta_min=1e-6)

    criterion = torch.nn.CrossEntropyLoss()
    logger = MetricLogger()

    for epoch in range(config['num_epoch']):
        net.train()
        header = 'Train Epoch {}:'.format(epoch)

        for images, labels, _ in logger.log_every(train_loader, 50, header=header):
            images, labels = images.to(args.device), labels.to(args.device)

            logits = net(normalize(images))
            loss = criterion(logits, labels)

            pred_idx = torch.argmax(logits.detach(), 1)
            correct = (pred_idx == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logger.meters['train_loss'].update(loss.item(), n=len(images))
            logger.meters['train_acc'].update(correct / len(images), n=len(images))

    net.eval()
    header = 'Test Epoch {}:'.format(epoch)
    for images, labels, _ in logger.log_every(test_loader, 50, header=header):
        images, labels = images.to(args.device), labels.to(args.device)

        with torch.no_grad():
            logits = net(normalize(images))
            loss = criterion(logits, labels)

        pred_idx = torch.argmax(logits.detach(), 1)
        correct = (pred_idx == labels).sum().item()

        logger.meters['test_loss'].update(loss.item(), n=len(images))
        logger.meters['test_acc'].update(correct / len(images), n=len(images))

    torch.save({'state_dict': net.state_dict()}, os.path.join(config['output_dir'], 'checkpoint.pth'))
    logger.clear()


def generate(args, config):
    normalize = normalize_list[config['normalize']]
    num_classes = config['dataset']['config']['num_classes']

    cluster = torch.load(config['cluster'], map_location='cpu')
    num_clusters = cluster['centers'].shape[0]

    noise = []
    for i in range(num_clusters):
        noise.append(torch.load(os.path.join(config['perturbation_dir'], f'perturbation_{i}.pth'), map_location='cpu')['perturbation'])
    noise = torch.cat(noise, dim=0)
    noise = torch.clamp(noise, -config['epsilon'] / 255., config['epsilon'] / 255)
    train_dataset = DataFolderWithClassNoise(config['dataset']['config']['train'], cluster['pred_idx'], noise=noise, resize_type=config['resize_type'])
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8)

    count = [0 for _ in range(config['dataset']['config']['num_classes'])]
    output_dir = config['ae_dir']
    print(output_dir)
    for i in range(len(count)):
        Path(os.path.join(output_dir, str(i))).mkdir(parents=True, exist_ok=True)
    print('Done floder')

    logger = MetricLogger()
    header = 'Generate cluster-wise UEs:'

    count = [0 for _ in range(num_classes)]
    for i in range(len(count)):
        Path(os.path.join(config['output_dir'], '..', 'ae', str(i))).mkdir(parents=True, exist_ok=True)

    for images, ground_truth, _ in train_loader:
        images_adv = images

        ground_truth = ground_truth.tolist()

        for i in range(len(images)):
            gt = ground_truth[i]
            save_image(images_adv[i], os.path.join(config['output_dir'], '..', 'ae', str(gt), f'{count[gt]}.png'))
            count[gt] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/stage_2.yaml')
    parser.add_argument('--experiment', '-e', type=str, default='uc_pets_cliprn50_rn18')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', type=int, default=2)
    args = parser.parse_args()
    yaml = YAML(typ='rt')
    with open(args.config, 'r') as f:
        config = yaml.load(f)[args.experiment]
    with open(config['data_config'], 'r') as f:
        data_config = yaml.load(f)[config['dataset']]
    
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)[args.experiment]
    # data_config = yaml.load(open(config['data_config'], 'r'), Loader=yaml.Loader)[config['dataset']]
    config['dataset'] = {'name': config['dataset'], 'config': data_config}
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

    if args.stage == 1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        yaml.dump(config, open(os.path.join(config['output_dir'], '..', 'config.yaml'), 'w+'))
        train_gnet(args, config)

        #world_size = torch.cuda.device_count()
        #mp.spawn(train_gnet, args=(world_size, args, config), nprocs=world_size, join=True)
    elif args.stage == 2:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        yaml.dump(config, open(os.path.join(config['output_dir'], 'config.yaml'), 'w+'))
        generate(args, config)
        train(args, config)
    elif args.stage == 3:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        yaml.dump(config, open(os.path.join(config['output_dir'], '..', 'config.yaml'), 'w+'))
        train_gnet_3d(args, config)
    elif args.stage == 4:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        yaml.dump(config, open(os.path.join(config['output_dir'], 'config.yaml'), 'w+'))
        generate(args, config)
        train(args, config)
    else:
        raise KeyError
