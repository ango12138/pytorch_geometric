"""Multi-node multi-GPU example on ogbn-papers100m.

Example way to run using srun:
srun -l -N<num_nodes> --ntasks-per-node=<ngpu_per_node> \
--container-name=cont --container-image=<image_url> \
--container-mounts=/ogb-papers100m/:/workspace/dataset
python3 path_to_script.py
"""
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GCN


def get_num_workers() -> int:
    num_workers = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_workers = len(os.sched_getaffinity(0)) // 2
        except Exception:
            pass
    if num_workers is None:
        num_workers = os.cpu_count() // 2
    return num_workers


def run(world_size, data, split_idx, model, acc):
    local_id = int(os.environ['LOCAL_RANK'])
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_id)
    device = torch.device(local_id)
    if rank == 0:
        print(f'Using {nprocs} GPUs...')

    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
    split_idx['valid'] = split_idx['valid'].split(
        split_idx['valid'].size(0) // world_size, dim=0)[rank].clone()
    split_idx['test'] = split_idx['test'].split(
        split_idx['test'].size(0) // world_size, dim=0)[rank].clone()

    model = DistributedDataParallel(model.to(device), device_ids=[local_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    kwargs = dict(
        data=data,
        batch_size=1024,
        num_workers=get_num_workers(),
        num_neighbors=[16, 16],
    )

    train_loader = NeighborLoader(
        input_nodes=split_idx['train'],
        shuffle=True,
        **kwargs,
    )
    val_loader = NeighborLoader(input_nodes=split_idx['valid'], **kwargs)
    test_loader = NeighborLoader(input_nodes=split_idx['test'], **kwargs)

    val_steps = 1000
    warmup_steps = 100
    acc = acc.to(rank)
    if rank == 0:
        print("Beginning training...")

    for epoch in range(1, 4):
        model.train()
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
                start = time.time()
            batch = batch.to(device)
            batch_size = batch.batch_size
            optimizer.zero_grad()
            y = batch.y[:batch_size].view(-1).to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch_size]
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            if rank == 0 and i % 10 == 0:
                print(f'Epoch: {epoch:02d}, Iteration: {i}, Loss: {loss:.4f}')

        dist.barrier()
        if rank == 0:
            sec_per_iter = (time.time() - start) / (i - warmup_steps)
            print(f"Avg Training Iteration Time: {sec_per_iter:.6f} s/iter")
        model.eval()
        acc_sum = 0.0
        for i, batch in enumerate(val_loader):
            if i >= val_steps:
                break
            if i == warmup_steps:
                start = time.time()

            batch = batch.to(device)
            batch_size = batch.batch_size
            with torch.no_grad():
                out = model(batch.x, batch.edge_index)[:batch_size]
            acc_sum += acc(out[:batch_size].softmax(dim=-1),
                           batch.y[:batch_size])
        acc_sum = torch.tensor(float(acc_sum), dtype=torch.float32,
                               device=rank)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        num_batches = torch.tensor(float(i), dtype=torch.float32,
                                   device=acc_sum.device)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(
                f"Validation Accuracy: {acc_sum/(num_batches) * 100.0:.4f}%", )
            sec_per_iter = (time.time() - start) / (i - warmup_steps)
            print(f"Avg Inference Iteration Time: {sec_per_iter:.6f} s/iter")
    dist.barrier()

    model.eval()
    acc_sum = 0.0
    for i, batch in enumerate(test_loader):
        batch = batch.to(device)
        batch_size = batch.batch_size
        with torch.no_grad():
            out = model(batch.x, batch.edge_index)[:batch_size]
        acc_sum += acc(out[:batch_size].softmax(dim=-1), batch.y[:batch_size])
    acc_sum = torch.tensor(float(acc_sum), dtype=torch.float32, device=rank)
    dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
    num_batches = torch.tensor(float(i), dtype=torch.float32,
                               device=acc_sum.device)
    dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"Test Accuracy: {acc_sum/(num_batches) * 100.0:.4f}%", )
    dist.barrier()


if __name__ == '__main__':
    # Setup multi-node:
    torch.distributed.init_process_group("nccl")
    nprocs = dist.get_world_size()
    assert dist.is_initialized(), "Distributed cluster not initialized"
    dataset = PygNodePropPredDataset(name='ogbn-papers100M',
                                     root='/datasets/ogb_datasets')
    split_idx = dataset.get_idx_split()
    model = GCN(dataset.num_features, 128, 2, dataset.num_classes)
    acc = Accuracy(task="multiclass", num_classes=dataset.num_classes)
    run(nprocs, dataset[0], split_idx, model, acc)
