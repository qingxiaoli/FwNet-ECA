import os
import argparse

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform

from my_dataset import MyDataSet
from model import FWNet_ECA_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
        

    tb_writer = SummaryWriter(log_dir=args.logdir)# tensorboard

    train_images_path, train_images_label,= read_split_data(args.data_path, mode="train")
    val_images_path, val_images_label,= read_split_data(args.data_path, mode="val")

    img_size = 224
    data_transform = {
        "train": transforms.Compose([#transforms.Resize(int(img_size * 1.143)),
                                     #transforms.CenterCrop(img_size),
                                     transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                     rand_augment_transform("rand-m9-mstd0.5-inc1", hparams={'translate_const': 100}),  # AutoAugment
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])


    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 4
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        #weights_dict = torch.load(args.weights, map_location=device)["model"] #torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        weights_dict = torch.load(args.weights, map_location=device)
        
        # del weights_dict["head.weight"] 
        '''             
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        '''
        
        print(model.load_state_dict(weights_dict, strict=False))
    
      
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除f_layer.complex_weight外,其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    
        
    
    
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    #optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-2)
    if args.if_scheduler:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.3, anneal_strategy='cos')
    else:
        scheduler = None

    weights_dir = args.weights_dir
    os.makedirs(weights_dir, exist_ok=True)  
    print("Saving weights to: ", weights_dir)  
    best_val_acc = 0.0  
    
    try:
        for epoch in range(args.epochs):
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    scheduler=scheduler,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)

            # validate
            val_loss, val_acc = evaluate(model=model,
                                        data_loader=val_loader,
                                        device=device,
                                        epoch=epoch)

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            
            if val_acc > best_val_acc:
                best_val_acc = val_acc  
                weights_path = os.path.join(weights_dir, "model-{}-train-{:.4f}-val-{:.4f}.pth".format(epoch, train_acc, val_acc))
                torch.save(model.state_dict(), weights_path)

    finally:
        tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=102)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.00005)


    parser.add_argument('--data-path', type=str,
                        default="E:\\flowers")

    # weights_dir
    parser.add_argument('--weights', type=str, default='C:\\Users\\Admin\\Desktop\\test_fwnet\\model-20-train-0.8275-val-0.9398.pth', help='the path of weights')
    parser.add_argument('--weights_dir', type=str, default='C:\\Users\\Admin\\Desktop\\test_fwnet', help='weights save dir')
    
    # tensorboard_dir
    parser.add_argument('--logdir', type=str, default='C:\\Users\\Admin\\Desktop\\test_fwnet', help='tensorboard log dir')
    

    parser.add_argument('--freeze-layers', type=bool, default=False, help='if freeze not head layers')
    parser.add_argument('--if_scheduler', type=bool, default=True, help='if use lr_scheduler')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
