"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


Modified version of Andrej Karpathy's "mingpt" repo found here: https://github.com/karpathy/minGPT
"""


import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from gpt_model import Character_GPT



class CharDataset(Dataset):
    def __init__(self, data):
        # Cấu hình kích thước khối (block size)
        self.block_size = 10

        # Xác định các ký tự có thể có và tạo ánh xạ giữa số và ký tự
        chars = sorted(list(set(data)))
        self.index2char = { i:ch for i,ch in enumerate(chars) }
        self.char2index = { ch:i for i,ch in enumerate(chars) }
        vocab_size = len(chars)
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Lấy một đoạn gồm (block_size + 1) ký tự từ dữ liệu
        chunk = self.data[idx:idx + self.block_size + 1]
        # Mã hóa mỗi ký tự thành một số nguyên
        dix = [self.char2index[s] for s in chunk]
        # Trả về dưới dạng các tensor
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:],dtype=torch.long)
        return x, y



def train_single_iteration(model, data_iter):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)
    batch = [t for t in batch]
    x, y = batch

    # Chạy forward mô hình
    loss = model.get_loss(x, y)

    # Lan truyền ngược và cập nhật tham số
    model.zero_grad(set_to_none=True)
    loss.backward()
   
    optimizer.step()

 
if __name__ == '__main__':
    # Các biến cấu hình
    learning_rate = 0.0004
    num_iterations = 10000
    batch_size = 500 
    checkpoint = 100 # Tần suất in kết quả của mô hình trong quá trình huấn luyện
    context = "Pacman" # Prompt dùng để sinh văn bản
    layer_size = 100
    n_layer = 6 # Số lượng khối transformer cần tạo


    # Xây dựng tập dữ liệu huấn luyện
    text = open('input.txt', 'r').read() 
    train_dataset = CharDataset(text)

    # Thiết lập dataloader
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=1,
    )

    train_iterations = iter(train_loader)

    # Thiết lập mô hình và optimizer
    model = Character_GPT(train_dataset.block_size, n_embd=layer_size, n_layer=n_layer, vocab_size=train_dataset.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)


    for i in range(num_iterations):
        train_single_iteration(model, train_iterations)
        
        if i % checkpoint == 0:
            with torch.no_grad():
                print("Iteration: " + str(i) + "\n")
                
                # Lấy mẫu từ mô hình...
                print("Prompt: " + context)
                print("Generated result: ")
                x = torch.tensor([train_dataset.char2index[s] for s in context])[None,...]
                y = model.generate(x, 500)[0]
                completion = ''.join([train_dataset.index2char[int(i)] for i in y])
                print(completion + "\n")
   