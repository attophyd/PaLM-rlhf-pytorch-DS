import deepspeed
import random
import tqdm
import numpy as np
import gzip

import torch
from lion_pytorch import Lion
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from palm_rlhf_pytorch import PaLM

# constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024


# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# instantiate palm
model = PaLM(
    num_tokens=256, 
    dim=512,
    depth=8, 
    flash_attn=True
).cuda()

# prepare enwik8 data
with gzip.open("data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len


train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# ds initialization & optimizer

num_devices = torch.cuda.device_count()
for rank in range(num_devices):
    device_name = torch.cuda.get_device_name(rank)
    print(f"GPU Device {rank}: {device_name}")

if num_devices == 1:
    print("Training initialization on single-GPU mode.")
elif num_devices > 1:
    print("Training initialization on multi-GPU mode.")
    deepspeed.init_distributed()
else:
    print("Local rank information not available. Training mode unknown.")

optim = Lion(model.palm_parameters(), lr=LEARNING_RATE)
model = model.to(torch.cuda.current_device())

deepspeed_config = {
    "train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATE_EVERY,
}

model_engine, optim, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optim,
    model_parameters=model.palm_parameters(),
    config_params=deepspeed_config,
)
print("Engine local rank: ", model_engine.local_rank)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model_engine.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(train_loader)
        batch = batch.to(
            torch.cuda.current_device()
        )
        loss = model_engine(batch, return_loss=True)
        model_engine.backward(loss)

    model_engine.step()
    model_engine.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model_engine.module.eval()
        with torch.no_grad():
            batch = next(val_loader)
            batch = batch.to(
                torch.cuda.current_device()
            )
            loss = model_engine.module(batch, return_loss=True)
            print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model_engine.module.eval()
        with torch.no_grad():
            inp = random.choice(val_dataset)[:PRIME_LENGTH]
            inp = inp.to(
                torch.cuda.current_device()
            )
            prime = decode_tokens(inp)
            print(f"{prime}\n{'*' * 100}")

            sample = model_engine.module.generate(GENERATE_LENGTH, inp[None, ...])
            output_str = decode_tokens(sample[0])
            print(output_str)
