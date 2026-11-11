import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from Bio import SeqIO
import numpy as np
import torchmetrics
import torch.optim as optim
import model as models
from logger import Logger
import time, os
from sklearn.model_selection import KFold

logger = Logger()


def log(str, log_out):
    print(str)
    logger.set_filename(log_out)
    logger.log(str + '\n')


if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Dictionary of Amino Acids and Numbers.
codeadict = {'A': "1", 'C': "2", 'D': "3", 'E': "4", 'F': "5", 'G': "6", 'H': "7", 'I': "8", 'K': "9", 'L': "10",
             'M': "11", 'N': "12", 'P': "13", 'Q': "14", 'R': "15", 'S': "16", 'T': "17", 'V': "18", 'W': "19", 'Y': "20"}


def format(predict_fasta):
    formatfasta = []
    recordlabel = []
    for record in SeqIO.parse(predict_fasta, 'fasta'):
        fastalist = []
        length = len(record.seq)
        if length <= 1000:
            for i in range(1, 1000-length+1):
                fastalist.append(0)
            for a in record.seq:
                fastalist.append(int(codeadict[a]))
        formatfasta.append(fastalist)
        if record.id.startswith('allergen'):
            recordlabel.append(1)
        else:
            recordlabel.append(0)
    inputarray = np.array(formatfasta)
    labelarray = np.array(recordlabel)
    return(inputarray, labelarray)
class EmbDataset(torch.utils.data.Dataset):
    def __init__(self, ids, labels, emb_dir, maxlen=1000, dim=1024):
        self.ids, self.labels = ids, labels
        self.emb_dir, self.maxlen, self.dim = emb_dir, maxlen, dim

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        a = np.load(os.path.join(self.emb_dir, self.ids[i] + '.npy'))   # (L, dim) fp16
        L = min(a.shape[0], self.maxlen)
        x = np.zeros((self.maxlen, self.dim), dtype=np.float32)
        x[self.maxlen - L:] = a[:L]        # LEFT pad — must match format()
        return torch.from_numpy(x), torch.tensor(self.labels[i], dtype=torch.float)


def read_ids(fasta):
    ids, labels = [], []
    for rec in SeqIO.parse(fasta, 'fasta'):
        ids.append(rec.id)
        labels.append(1.0 if rec.id.startswith('allergen') else 0.0)
    return ids, labels


def validation(model, valid_ds, criterion, args):
    valid_loader = DataLoader(
        dataset=valid_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model.eval()
    accuracy = torchmetrics.Accuracy(task='binary').to(device)
    recall = torchmetrics.Recall(task='binary').to(device)
    precision = torchmetrics.Precision(task='binary').to(device)
    auroc = torchmetrics.AUROC(task='binary').to(device)
    f1 = torchmetrics.F1Score(task='binary').to(device)

    finaloutputs = torch.tensor([]).to(device)
    finallabels = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        valid_loss = 0.0
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            labels = torch.as_tensor(labels, dtype=torch.long)
            finaloutputs = torch.cat([finaloutputs, outputs], 0)
            finallabels = torch.cat([finallabels, labels], 0)
        accuracy(finaloutputs, finallabels)
        recall(finaloutputs, finallabels)
        precision(finaloutputs, finallabels)
        auroc(finaloutputs, finallabels)
        f1(finaloutputs, finallabels)
        accuracy_value = accuracy.compute()
        recall_value = recall.compute()
        precision_value = precision.compute()
        auroc_value = auroc.compute()
        f1_value = f1.compute()
        accuracy.reset()
        recall.reset()
        precision.reset()
        auroc.reset()
        f1.reset()
    return (valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value)


def train(args):
    tag = args.tag or args.model
    os.makedirs('weights', exist_ok=True)

    if args.embeddings:
        ids, labels = read_ids(args.inputs)
        full = EmbDataset(ids, labels, args.embeddings, 1000, args.in_channels)
    else:
        xa, ya = format(args.inputs)
        full = TensorDataset(torch.tensor(xa, dtype=torch.long),
                             torch.tensor(ya, dtype=torch.float))
    valid_loss_sum, accuracy_sum, recall_sum, precision_sum, auroc_sum, f1_sum = 0, 0, 0, 0, 0, 0
    k = 0
    skf = KFold(n_splits=10, shuffle=False)
    for train_index, valid_index in skf.split(np.arange(len(full))):
        train_ds = Subset(full, train_index)
        valid_ds = Subset(full, valid_index)
        k = k+1
        model = getattr(models, args.model)(in_channels=args.in_channels,
                                            onehot=args.embeddings is None)
        model.to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
        criterion = nn.BCELoss().to(device)
        train_loader = DataLoader(
            dataset=train_ds, batch_size=args.batch_size, shuffle=True,
            drop_last=True, num_workers=2)
        best_f1 = 0
        best_metrics = None
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                log('[k: %d, batch: %d] train_loss: %.3f' %
                    (k, i + 1, loss.item()), 'train.log')
                now = time.asctime(time.localtime(time.time()))
            valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value = validation(
                model, valid_ds=valid_ds, criterion=criterion, args=args)
            # Generate validation results for each epoch.
            log('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f' % (epoch, valid_loss, accuracy_value, recall_value,
                precision_value, auroc_value, f1_value), f'./{tag}.{k}.fold.everyepoch.valid.txt')
            # Save the model with the largest F1 value.
            if f1_value > best_f1:
                best_f1 = f1_value
                torch.save(model.state_dict(), f'weights/{tag}.{k}.pt')
                best_metrics = (valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value)
        # Generate validation results for each fold
        if best_metrics is not None:
            valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value = best_metrics

        log('[k: %d] valid_loss: %.3f accuracy_value: %.6f recall_value: %.6f precision_value: %.6f auroc_value: %.6f f1_value: %.6f' % (
            k, valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value), 'valid.log')
        valid_loss_sum += valid_loss
        accuracy_sum += accuracy_value.item()
        recall_sum += recall_value.item()
        precision_sum += precision_value.item()
        auroc_sum += auroc_value.item()
        f1_sum += f1_value.item()
        if k >= args.folds:
            break
    log('valid_loss: %.3f accuracy_value: %.6f recall_value: %.6f precision_value: %.6f auroc_value: %.6f f1_value: %.6f' % (
        valid_loss_sum/k, accuracy_sum/k, recall_sum/k, precision_sum/k, auroc_sum/k, f1_sum/k), 'valid.log')
    now = time.asctime(time.localtime(time.time()))
    log(str(now), 'train.log')
