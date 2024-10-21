import os
import sys
import json
import torch
from tqdm import tqdm


def read_split_data(root: str, mode: str = 'train', supported = [".jpg", ".JPG", ".png", ".PNG", ".JPEG"]):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    assert mode in ['train', 'val'], "mode must be 'train' or 'val'."

    dataset_dir = os.path.join(root, mode)

    dataset_class = [cla for cla in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, cla))]

    dataset_class.sort()
    

    class_indices = dict((k, v) for v, k in enumerate(dataset_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    images_path = []  
    images_label = [] 
    every_class_num = []  


    for cla in dataset_class:
        cla_path = os.path.join(dataset_dir, cla)

        images = [os.path.join(dataset_dir, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
 
        images.sort()

        image_class = class_indices[cla]
  
        every_class_num.append(len(images))


        for img_path in images:
            images_path.append(img_path)
            images_label.append(image_class)

    print("{} images were found in the {} dataset.".format(sum(every_class_num), mode))

    return images_path, images_label


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler=None):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  
    accu_num = torch.zeros(1).to(device)   
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]


        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   
    accu_loss = torch.zeros(1).to(device)  

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
