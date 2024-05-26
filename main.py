import argparse
import torch
import dataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dataset import Shakespeare
from model import CharRNN, CharLSTM
from generate import generate
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

# import some packages you need here

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()
model_name = args.model

EPOCHS = 100

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    # write your codes here
    trn_loss = []

    model.train()
    batch_losses = []
    for i, (inputs, labels) in enumerate(trn_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        hidden = model.init_hidden(inputs.size(0))

        if model_name == 'RNN' :
            hidden = hidden.detach()

        model.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output.view(-1, model.output_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    trn_loss = np.average(batch_losses)

    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    # write your codes here
    model.eval()
    val_loss = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            hidden = model.init_hidden(inputs.size(0))
            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(-1, model.output_size), labels.view(-1))
            val_loss.append(loss.item())
    val_loss = np.mean(val_loss)

    return val_loss



def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 64
    embedding_dim = 128
    hidden_dim = 128
    n_layers = 1

    dataset = Shakespeare('shakespeare_train.txt')
    indices = list(range(len(dataset)))

    split = int(np.floor(0.2*len(dataset)))   # 8:2 split
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size, train_sampler)
    test_loader = DataLoader(dataset, batch_size, valid_sampler)

    vocab_size = len(dataset.char)

    if model_name == 'RNN' :
        model = CharRNN(vocab_size, embedding_dim, hidden_dim, vocab_size, n_layers)
    if model_name == 'LSTM' :
        model = CharLSTM(vocab_size, embedding_dim, hidden_dim, vocab_size, n_layers)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(DEVICE)



    print(f" ========== 📍 Applied Model : {model_name} ========== ")

    train_loss, valid_loss = [], []
    best_loss = float('inf')
    for epoch in range(EPOCHS) :

        trn_loss = train(model, train_loader, DEVICE, criterion, optimizer)
        vld_loss = validate(model, test_loader, DEVICE, criterion)

        print(f'EPOCH({epoch})   Train Loss: {trn_loss:4f}   Test Loss: {vld_loss:4f}')

        train_loss.append(trn_loss)
        valid_loss.append(vld_loss)

        if vld_loss < best_loss :
            best_loss = vld_loss
            best_epoch = epoch
            best_params = model.state_dict()

    pickle.dump({'train_loss' : train_loss,
                'test_loss' : valid_loss}, open(f"{model_name}_loss.pickle", "wb"))
    torch.save(best_params, f'models/best_state_{model_name}_epoch{best_epoch}.pth')
    print(f'Best Model Saved at (epoch{best_epoch})')


    # ====================== text generated with best model ====================

    model.load_state_dict(torch.load(f'models/best_state_{model_name}_epoch{best_epoch}.pth'))
    
    seed_chars = ['QUEEN ELIZABETH:', 'My name is', 'First', 'I can', 'Here is']
    for t in (0.1, 0.5, 1, 3, 10) :
        generated_texts = ''
        for seed in seed_chars :
            generated_text = generate(model, seed, t, 1000, dataset, DEVICE)
            generated_texts += generated_text

            print(f' ========== GENERATED TEXT start with "{seed}" ==========')
            print(generated_text)

        with open(f'generated/shakespeare_by_{model_name}_T={t}.txt', 'w') as f:
            f.write(generated_texts)

if __name__ == '__main__':
    main()
    