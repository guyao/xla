from typing import List
import torch
import torch.distributed as dist
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
  # gm.graph.print_tabular()
  return gm.forward


def dummy_collective_fn_xm(input):
  res_tensor = xm.all_reduce(xm.REDUCE_SUM, input)
  res_tensor += 3.0
  res_tensor = xm.all_gather(res_tensor, dim=0)
  return res_tensor


def dummy_collective_fn_dist(input, async_op):
  # TODO(JackCaoG, zpcore): fix the issue dist.all_reduce issue with openxla backend
  reduce_tensor = dist.all_reduce(input, dist.ReduceOp.SUM, async_op=async_op)
  output_tensor = torch.empty((1, xr.world_size()))
  dist.all_gather_into_tensor(output_tensor, input, None, async_op=async_op)
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
      res_tensor = dummy_collective_fn_dist(ordinal_tensor, async_op=True)
    elif test_mode == 'dynamo':
      compiled_collective = torch.compile(
          dummy_collective_fn_dist, backend=my_compiler)
      res_tensor = compiled_collective(ordinal_tensor, async_op=False)
    print(res_tensor)
    expected_tensor = torch.tensor(
        [world_size * world_size / 2] * world_size, dtype=torch.float)
    torch_xla.sync()
    torch.allclose(res_tensor.cpu(), expected_tensor)
    # assert met.metric_data("ExecuteTime")[0] == 1


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
  torch_xla.launch(_mp_fn_dist, args=())
