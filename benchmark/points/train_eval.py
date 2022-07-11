import time

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.profiler import ProfilerActivity, profile

from torch_geometric.loader import DataLoader
from torch_geometric.profile import trace_handler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_train(train_dataset, test_dataset, model, epochs, batch_size, lr,
              lr_decay_factor, lr_decay_step_size, weight_decay):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        print("Epoch {} starts".format(epoch))
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        train(model, optimizer, train_loader, device)
        test_acc = test(model, test_loader, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}, '
              f'Duration: {t_end - t_start:.2f}')

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']


def run_inference(test_dataset, model, epochs, batch_size, profiling):
    model = model.to(device)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        print("Epoch {} starts".format(epoch))
        if epoch == epochs:
            if profiling:
                with profile(
                        activities=[
                            ProfilerActivity.CPU, ProfilerActivity.CUDA
                        ], on_trace_ready=trace_handler) as p:
                    inference(model, test_loader, device)
                    p.step()
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_start = time.time()

                inference(model, test_loader, device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_end = time.time()
                duration = t_end - t_start
                print("End-to-End time: {} s".format(duration), flush=True)
        else:
            inference(model, test_loader, device)


def run(train_dataset, test_dataset, model, epochs, batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay, inference,
        profiling):
    if not inference:
        run_train(train_dataset, test_dataset, model, epochs, batch_size, lr,
                  lr_decay_factor, lr_decay_step_size, weight_decay)
    else:
        run_inference(test_dataset, model, epochs, batch_size, profiling)


def train(model, optimizer, train_loader, device):
    model.train()

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()


def test(model, test_loader, device):
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data.pos, data.batch).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    test_acc = correct / len(test_loader.dataset)

    return test_acc


@torch.no_grad()
def inference(model, test_loader, device):
    model.eval()
    for data in test_loader:
        data = data.to(device)
        model(data.pos, data.batch)
