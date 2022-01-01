import numpy as np
from functools import partial
import scipy.io as scio
import h5py
import torch
from Model_define_pytorch import AutoEncoder, DatasetFolder, MyLoss, DatasetFolderTrain
import os
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu)


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def set_seed(seed):
    os.environ["PYTHONASHSEED"] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def lr_scheduler(step, warm_up_step, max_step):
    if step < warm_up_step:
        return 1e-2 + (1 - 1e-2) * step / warm_up_step
    return 1e-2 + (1 - 1e-2) * 0.5 * (1 + math.cos((step - warm_up_step) / (max_step - warm_up_step) * math.pi))
    
        
class DefaultCfg:
    seed = 1992
    batch_size = 128
    epochs = 1000
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_workers = 4
    feedback_bits = 512
    save_dir = "./Modelsave"


def read_data(data_load_address='../train'):
    mat = scio.loadmat(data_load_address+'/Htrain.mat')
    x_train = mat['H_train']  # shape=8000*126*128*2
    
    x_train = np.transpose(x_train.astype('float32'),[0,3,1,2])
    print(np.shape(x_train))
    mat = scio.loadmat(data_load_address+'/Htest.mat')
    x_test = mat['H_test']  # shape=2000*126*128*2
    
    x_test = np.transpose(x_test.astype('float32'),[0,3,1,2])
    print(np.shape(x_test))
    return x_train, x_test
    
    
def validate(best_loss, test_loader, model, criterion, save_dir, prefix=""):
    modelSave1 = os.path.join(save_dir, prefix + "encoder.pth.tar")
    modelSave2 = os.path.join(save_dir, prefix + "decoder.pth.tar")
    total_loss = 0
    nums = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            input = input.cuda()
            output = model(input)
            total_loss += criterion(output, input).item() * input.size(0)
            nums += input.size(0)
        average_loss = total_loss / nums
        if average_loss < best_loss:
            torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
            torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            print("Model saved")
            best_loss = average_loss
    return best_loss
    
    
def train(cfg, x_train, x_test):
    os.makedirs(cfg.save_dir, exist_ok=True)
    train_dataset = DatasetFolderTrain(x_train, training=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    test_dataset = DatasetFolder(x_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True)
    
    model = AutoEncoder(cfg.feedback_bits)
    model.to(device)
    criterion = MyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=cfg.learning_rate, 
                                  weight_decay=cfg.weight_decay)
    max_step = len(cfg.x_train) // cfg.batch_size * cfg.epochs
    scheduler = LambdaLR(optimizer=optimizer, 
                         lr_lambda=partial(lr_scheduler, 
                                           warm_up_step=cfg.warmup, 
                                           max_step=max_step))
    ema = EMA(model, 0.995)
    ema.register()
    ema_start = True
    
    best_loss = 1
    ema_best_loss = 1
    for epoch in range(epochs):
        model.train()
        for i, input in enumerate(train_loader):
            input = input.to(device)
            output = model(input)
            loss = criterion(output, input)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            scheduler.step()
            if ema_start:
                ema.update()
                
        model.eval()
        best_loss = validate(best_loss, test_loader, model, criterion, cfg.save_dir, prefix="")
        if ema_start:
            ema.apply_shadow()
            ema_best_loss = validate(ema_best_loss, test_loader, model, criterion, cfg.save_dir, prefix="ema_")
            ema.restore()
            

    
def main():
    x_train, x_test = read_data('../train')
    cfg = DefaultCfg()
    train(cfg, x_train, x_test)
    
    
if __name__ == '__main__':
    main()
