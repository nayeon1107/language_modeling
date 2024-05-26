# import some packages you need here
import torch
import torch.nn.functional as F
import numpy as np

from dataset import Shakespeare

def generate(model, seed_characters, temperature, length, dataset, device, *args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """

    # write your codes here
    model.eval()
    input_seq = torch.tensor([dataset.char2idx[s] for s in seed_characters]).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    samples = seed_characters

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            output = output[-1, :] / temperature
            prob = F.softmax(output, dim=0).data
            top_char = torch.multinomial(prob, 1).item()
            samples += dataset.idx2char[top_char]

            input_seq = torch.tensor([[top_char]], dtype=torch.long).to(device)
    samples += '\n'
    
    return samples