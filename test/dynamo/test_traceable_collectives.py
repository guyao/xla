from typing import List
import torch
import torch.distributed as dist
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met


def dummy_collective_fn_xm(input):
  res_tensor = xm.all_reduce(xm.REDUCE_SUM, input)
  res_tensor += 3.0
  res_tensor = xm.all_gather(res_tensor, dim=0)
  return res_tensor


def dummy_collective_fn_dist(input):
  # zpcore: We can't use the return value from dist.xxx. The result value will be a `work` object from
  # the dist function call: https://github.com/pytorch/pytorch/blob/018e48c337094c8800483f1e577a1ec241982001/torch/distributed/distributed_c10d.py#L2501-L2506
  dist.all_reduce(input, dist.ReduceOp.SUM)  # input update in place
  input += 3.0
  output_tensor = torch.empty((1, xr.world_size()))
  dist.all_gather_into_tensor(output_tensor, input, None)
  return output_tensor


def collective_broadcast_and_cos(input, src):
  res_tensor = torch.ops._c10d_functional.broadcast(input, src, "default")
  return torch.cos(res_tensor)


def _mp_fn(index):
  device = xm.xla_device()
  world_size = xr.world_size()
  if xm.xla_device_hw(device) not in ('TPU', 'CUDA', 'NEURON'):
    print(f'skip this test for hw {xm.xla_device_hw(device)}')
    return
  ordinal_tensor = torch.tensor([index], dtype=torch.float).to(device)
  for dynamic in [True, False]:
    met.clear_all()
    compiled_collective = torch.compile(
        dummy_collective_fn_xm, backend="openxla", dynamic=dynamic)
    res_tensor = compiled_collective(ordinal_tensor)
    expected_tensor = torch.tensor(
        [world_size * world_size / 2] * world_size, dtype=torch.float) + 3.0
    torch_xla.sync()
    torch.allclose(res_tensor.cpu(), expected_tensor)
    assert met.metric_data("ExecuteTime")[0] == 1

  for dynamic in [True, False]:
    met.clear_all()
    compiled_collective = torch.compile(
        collective_broadcast_and_cos, backend="openxla", dynamic=dynamic)
    res_tensor = compiled_collective(ordinal_tensor, 2)
    expected_tensor = torch.cos(torch.tensor([2]))
    torch_xla.sync()
    torch.allclose(res_tensor.cpu(), expected_tensor)
    # TODO (JackCaoG): fix CI issue and uncomment below.
    # assert met.metric_data("ExecuteTime")[0] == 1


def _mp_fn_dist(index):
  dist.init_process_group("xla", init_method='xla://')
  device = xm.xla_device()
  world_size = xr.world_size()
  for test_mode in ['eager', 'dynamo']:
    if xm.xla_device_hw(device) not in ('TPU', 'CUDA', 'NEURON'):
      print(f'skip this test for hw {xm.xla_device_hw(device)}')
      return
    ordinal_tensor = torch.tensor([index], dtype=torch.float).to(device)
    met.clear_all()
    if test_mode == 'eager':
      # no need to invoke works' wait operation in torch.
      res_tensor = dummy_collective_fn_dist(ordinal_tensor)
    elif test_mode == 'dynamo':
      compiled_collective = torch.compile(
          dummy_collective_fn_dist, backend='openxla')
      res_tensor = compiled_collective(ordinal_tensor)
    expected_tensor = torch.tensor(
        [world_size * world_size / 2] * world_size, dtype=torch.float) + 3.0
    torch_xla.sync()
    torch.allclose(res_tensor.cpu(), expected_tensor)
    # TODO (JackCaoG): fix CI issue and uncomment below.
    # assert met.metric_data("ExecuteTime")[0] == 1


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
  torch_xla.launch(_mp_fn_dist, args=(), debug_single_process=False)
