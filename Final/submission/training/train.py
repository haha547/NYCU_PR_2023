from dataset import Synth90kDataset, synth90k_collate_fn
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from model import CRNN
import torch.optim as optim
from torch.nn import CTCLoss
import torchvision.transforms as transforms
from evaluate import evaluate
import os
from config import train_config as config
import numpy as np


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5) # gradient clipping with 5
    optimizer.step()
    return loss.item()

def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    lr = config['lr']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transform = transforms.Compose([
    #     transforms.RandomRotation(5),
    #     transforms.RandomVerticalFlip(p = 0.3),
    # ])

    dataset = Synth90kDataset("train", mode= "train", transform=None)

    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

    train_dl = DataLoader(
        dataset = train_set,
        batch_size = train_batch_size,
        shuffle = True,
        num_workers = 0,
        collate_fn=synth90k_collate_fn)
    val_dl = DataLoader(
        dataset = valid_set,
        batch_size = eval_batch_size,
        shuffle = False,
        num_workers = 0,
        collate_fn=synth90k_collate_fn)
    num_class = len(Synth90kDataset.LABEL2CHAR) + 1

    crnn = CRNN(1, 32, 100, num_class,
                map_to_seq_hidden= 64,
                rnn_hidden= 256,
                leaky_relu= False).cuda()

    optimizer = optim.RMSprop(crnn.parameters(), lr= lr)
    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    criterion.cuda()
    # sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, 200, 0.5)
    i = 1

    hist = dict(
                loss=np.zeros((epochs, )), val_loss=np.zeros((epochs, )),
                val_acc=np.zeros((epochs, ))
            )
    
    for epoch in range(epochs):
        print(f'epoch = {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0

        for train_data in train_dl:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            # if i % show_interval == 0:
            #     print(f'train_batch_loss[{i}]: , {(loss / train_size):.2e}')

            if i % valid_interval == 0:

                evaluation = evaluate(crnn, val_dl, criterion,
                                      decode_method= "beam_search",
                                      beam_size= 10)
                print('valid_evaluation: loss={loss:.2e}, acc={acc:.2f}'.format(**evaluation))
                
                if i % save_interval == 0:
                    prefix = 'crnn'
                    acc = evaluation['acc']
                    save_model_path = os.path.join('checkpoints/',
                                                   f'{prefix}_{i:06}_loss{acc:-.3f}.pt')
                    torch.save(crnn.state_dict(), save_model_path)
                    print('save model at ', save_model_path)
            i += 1
        # hist["loss"][epoch] = tot_train_loss / tot_train_count
        # evaluation = evaluate(crnn, val_dl, criterion,
        #                               decode_method= "beam_search",
        #                               beam_size= 10)
        # hist["val_loss"][epoch] = evaluation["loss"]
        # hist["val_acc"][epoch] = evaluation["acc"] 
        # sche_fn.step()
        print(f'train_loss: {(loss / train_size):.2e}')



if __name__ == '__main__':
    main()