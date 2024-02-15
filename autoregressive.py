import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class AutoregressiveWrapper(nn.Module):
    def __init__(self, gpt_model, max_seq_len=1024, device=None) -> None:
        super().__init__()
        self.model = gpt_model.to(device)
        self.device = device
        if self.model.vocab_size is not None:
            self.max_seq_len = self.model.vocab_size
        else:
            self.max_seq_len = max_seq_len
    
    def forward(self, question: torch.Tensor, answer: torch.Tensor, criterion):
        
        answer_input = answer[:, :-1]
        answer_target = answer[:, 1:]

        # create masks and add dimensions
        question_mask, answer_input_mask, answer_target_mask = create_masks(question, answer_input, answer_target)
        output = self.model(question, question_mask, answer_input, answer_input_mask)

        loss = criterion(output, answer_target, answer_target_mask)
        return output, loss


def generate(
    transformer,
    question,
    max_generate_len,
    word_map
):
    '''
    Performs Greedy decoding with a batch size of 1
    '''
    device = 'cuda'
    transformer.eval()

    reverse_word_map = {v: k for k, v in word_map.items()}
    start_token = 2

    enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split(' ')]
    # enc_qus = pad_question(word_map, question, 50)
    question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
    question_mask = (question != 0).unsqueeze(1).unsqueeze(1)

    encoded = transformer.encode_stack(question, question_mask)
    words = torch.LongTensor([[start_token]]).to(device)

    for _ in range(max_generate_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0,1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode_stack(words, target_mask, encoded, question_mask)
        
        # predictions_probs = transformer.logits(decoded[:, -1])
        predictions_probs = transformer.logits(decoded[:, -1])
        _, next_word = torch.max(predictions_probs, dim=1)
        next_word = next_word.item()

        if next_word == word_map['<eos>']:
            break
        
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim=1)
    
    if words.dim() == 2:
        words = words.squeeze(0).tolist()
    
    sentence_tokens = [w for w in words if w not in {word_map['<sos>']}]
    sentence = ' '.join([reverse_word_map[sentence_tokens[k]] for k in range(len(sentence_tokens))])

    return sentence


def pad_question(word_map, sentence, max_len):
    words = sentence.split(' ')[:max_len]
    enc_c = [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<pad>']] * (max_len - len(words))
    return enc_c


with open('word_map.json', 'r') as j:
    word_map = json.load(j)

checkpoint = torch.load('checkpoints/checkpoint_9.pth.tar')
model = checkpoint['transformer']

while(1):
    question = input("\n🍊 > ")
    if question == 'quit' or question == 'q':
        break
    
    max_len = 200
    sentence = generate(model.model, question, int(max_len), word_map=word_map)
    print("✨:: ", sentence)

    # What is the family name of raspberry?