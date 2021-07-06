import os
import argparse
import torch
from transformers import BertTokenizer
from tensorboardX import SummaryWriter


from model import *
from data import Dataset, collate_fn
from utils import clip_gradient

def train(model, dataloader, optimizer, device, epoch, writer):
    model.train()

    for idx, (positive, negative, mask_p, mask_n, id_p, id_n) in enumerate(dataloader):
        positive = positive.to(device)
        negative = negative.to(device)
        mask_p = mask_p.to(device)
        mask_n = mask_n.to(device)
        id_p = id_p.to(device)
        id_n = id_n.to(device)

        optimizer.zero_grad()
        score_positive = net(positive, mask_p, id_p)
        score_negative = net(negative, mask_n, id_n)

        loss = criterion(score_positive, score_negative)
        loss.backward()
        clip_gradient(optimizer, grad_clip)
        optimizer.step()
        
        writer.add_scalar('train loss', loss.item(), epoch * len(dataloader) + idx)

def val(model, dataloader, device, epoch, writer):
    model.eval()
    losses = []

    for idx, (positive, negative, mask_p, mask_n, id_p, id_n) in enumerate(dataloader):
        positive = positive.to(device)
        negative = negative.to(device)
        mask_p = mask_p.to(device)
        mask_n = mask_n.to(device)
        id_p = id_p.to(device)
        id_n = id_n.to(device)

        optimizer.zero_grad()
        score_positive = net(positive, mask_p, id_p)
        score_negative = net(negative, mask_n, id_n)

        loss = criterion(score_positive, score_negative)
        losses.append(loss.item())
        
    writer.add_scalar('val loss', sum(losses) / len(losses), epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type = str, default = 'data/train')
    parser.add_argument('--val', type = str, default = 'data/val')
    parser.add_argument('--ckpt', type = str, default = 'ckpt')
    parser.add_argument('--logdir', type = str, default = 'runs')
    parser.add_argument('--batch', type = int, default = 32)
    parser.add_argument('--epoch', type = int, default = 1)
    parser.add_argument('--begin', type = int, default = 0)
    parser.add_argument('--lr', type = float, default = 1e-6)
    parser.add_argument('--bert', type = str, default = 'scibert_scivocab_uncased')

    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        os.makedirs(args.ckpt)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(args.logdir)
    grad_clip = 5

    #dataloader
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    dataset_train = Dataset(args.train, tokenizer)
    dataloader_train = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = args.batch, shuffle = True, collate_fn = collate_fn)
    dataset_val = Dataset(args.val, tokenizer)
    dataloader_val = torch.utils.data.DataLoader(dataset = dataset_val, batch_size = 1, shuffle = False, collate_fn = collate_fn)

    #network
    net = BERT_pair(args.bert)
    net = net.to(device)
    net = torch.nn.DataParallel(net)

    #loss and optimizer
    criterion = TripletLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)

    #initial validation
    with torch.no_grad():
        val(net, dataloader_val, device, 0, writer)

    for epoch in range(args.begin, args.epoch):
        train(net, dataloader_train, optimizer, device, epoch, writer)
        with torch.no_grad():
            val(net, dataloader_val, device, epoch + 1, writer)

        torch.save(net.state_dict(), os.path.join(args.ckpt, 'model_epoch{}.pth'.format(epoch)))

    writer.close()