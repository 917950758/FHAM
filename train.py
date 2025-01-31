import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.FHAMN import FHAMN
from nets.FHAMN_training import weights_init
from utils.callbacks import LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils_fit import fit_one_epoch

torch.cuda.current_device()
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    # -------------------------------#
    Cuda = True
    # -------------------------------#
    #   需要的分类个数+1，如2+1
    # -------------------------------#
    num_classes = 8

    # -------------------------------------------------------------------#
    #   所使用的的主干网络：
    #   mobilenet、xception
    # -------------------------------------------------------------------#
    backbone = "mobilenet"
    # -------------------------------------------------------------------#
    #   所使用的注意力机制：
    #   CBAM、
    #   所使用的ASPP：
    #   res、null
    # -------------------------------------------------------------------#
    att_name = 'FHAM'
    aspp_name = 'res'
    # ---------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重。
    # ---------------------------------------------------------------------------------------------#
    pretrained = True
    model_path = r"model_data/deeplab_mobilenetv2.pth"
    # ---------------------------------------------------------#
    #   下采样的倍数8、16 
    #   8下采样的倍数较小、理论上效果更好，但也要求更大的显存
    # ---------------------------------------------------------#
    downsample_factor = 16
    # ------------------------------#
    #   输入图片的大小
    # ------------------------------#
    input_shape = [1024, 1024]
    # ----------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结
    # ----------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 30
    Freeze_batch_size = 4
    Freeze_lr = 3e-4
    # ----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    # ----------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr = 3e-4
    # ------------------------------#
    #   数据集路径
    # ------------------------------#
    VOCdevkit_path = r'Urban'
    #   每2轮保存一次
    save_period = 2
    # ---------------------------------------------------------------------#
    #   是否使用diceloss
    # ---------------------------------------------------------------------#
    dice_loss = True
    # ---------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    # ---------------------------------------------------------------------#
    focal_loss = False

    # ---------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    # ---------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------#
    Freeze_Train = True
    # ------------------------------------------------------#
    #   用于设置使用多线程读取数据
    # ------------------------------------------------------#
    num_workers = 4

    model = FHAMN(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                   pretrained=pretrained, att_name=att_name, aspp_name=aspp_name)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history = LossHistory("ISPRS")

    # ----------------------------------------- -------------#
    #   训练数据集
    # ------------------------------------------------------#
    with open(os.path.join(VOCdevkit_path, "train.txt"), "r") as f:
        train_lines = f.readlines()

    # ------------------------------------------------------#
    #   验证数据集
    # ------------------------------------------------------#
    with open(os.path.join(VOCdevkit_path, "val.txt"), "r") as f:
        val_lines = f.readlines()

    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        epoch_step = len(train_lines) // batch_size
        epoch_step_val = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=0)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)

        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights,
                          num_classes,save_period)
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        epoch_step = len(train_lines) // batch_size
        epoch_step_val = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=0)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights,
                          num_classes,save_period)
            lr_scheduler.step()

