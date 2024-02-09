import json
import torch
from torch.utils.data import DataLoader
from model import *
from utils import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(AlpacaDataset(),
                          batch_size=24,
                          shuffle=True,
                          pin_memory=True)

d_model = 512
n_heads = 8
n_layers = 6
epochs = 10

with open('word_map.json', 'r') as j:
    word_map = json.load(j)

model = Transformer(
            d_model=d_model, 
            n_heads=n_heads, 
            n_layers=n_layers, 
            word_map=word_map
        ).to(DEVICE)

adam_optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
model_optimizer = AdamWarmup(model_size=d_model, warmup_steps=4000, optimizer=adam_optimizer)
loss_criterion = LossWithLS(len(word_map), 0.1)

def train(train_loader, model, criterion, epochs):
    model.train()
    sum_loss = 0
    count = 0

    for epoch in range(epochs):
        for i, (question, answer) in enumerate(train_loader):
            samples = question.shape[0]

            # Use GPU if available
            question = question.to(DEVICE)
            answer = answer.to(DEVICE)

            answer_input = answer[:, :-1]
            answer_target = answer[:, 1:]

            # create masks and add dimensions
            question_mask, answer_input_mask, answer_target_mask = create_masks(question, answer_input, answer_target)
            
            # get transformer outputs
            out = model(question, question_mask, answer_input, answer_input_mask)

            loss = criterion(out, answer_target, answer_target_mask)

            # backpropagation
            model_optimizer.optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            sum_loss += loss.item() * samples
            count += samples

            if i % 100 == 0:
                print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss/count))

        state = {'epoch': epoch, 'transformer': model, 'transformer_optimizer': model_optimizer}
        torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')

train(train_loader, model, loss_criterion, 10)