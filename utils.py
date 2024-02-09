import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlpacaDataset(Dataset):
    def __init__(
        self
    ) -> None:
        self.pairs = json.load(open('pairs_encoded.json'))
        self.size = len(self.pairs)
    
    def __getitem__(self, index):
        question = torch.LongTensor(self.pairs[index][0])
        answer = torch.LongTensor(self.pairs[index][1])

        return question, answer
    
    def __len__(self):
        return self.size


class AdamWarmup:
    def __init__(self, model_size, warmup_steps: int, optimizer) -> None:
        self.lr = 0
        self.current_step = 0
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_steps = warmup_steps

    def get_lr(self) -> float:
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # update the learning rate
        self.lr = lr
        self.optimizer.step()


def create_masks(question, answer_input, answer_target):
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0,1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)

    question_mask = (question != 0).to(device)
    question_mask = question_mask.unsqueeze(1).unsqueeze(1)     # (batch size, 1, 1, max words)

    answer_input_mask = answer_input != 0
    answer_input_mask = answer_input_mask.unsqueeze(1)  # (batch size, 1, max words)
    answer_input_mask = answer_input_mask & subsequent_mask(answer_input.size(-1)).type_as(answer_input_mask.data)
    answer_input_mask = answer_input_mask.unsqueeze(1)  # (batch size, 1, max words, max words)

    answer_target_mask = answer_target != 0             # (batch size, max words)

    return question_mask, answer_input_mask, answer_target_mask


class LossWithLS(nn.Module):
    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size
    
    def forward(self, prediction, target, mask):
        """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
        prediction = prediction.view(-1, prediction.size(-1))   # (batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1)   # (batch_size * max_words)
        mask = mask.float()
        mask = mask.view(-1)       # (batch_size * max_words)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)    # (batch_size * max_words, vocab_size)
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        return loss