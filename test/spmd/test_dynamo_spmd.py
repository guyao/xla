import os
import sys

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.debug.metrics as met
import torch_xla.experimental.dynamo_mark_sharding
import unittest

import test_xla_sharding_base


class SimpleLinear(nn.Module):

  def __init__(self, mesh=None):
    super(SimpleLinear, self).__init__()
    self.fc1 = nn.Linear(128, 128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, 1)
    # Add an additional 1x1 layer at the end to ensure the final layer
    # is not sharded.
    self.fc3 = nn.Linear(1, 1)
    # If mesh is not none, we'll do a mark sharding inside the forward function
    # to ensure dynamo can recognize and trace it in a torch compile.
    self.mesh = mesh

  def forward(self, x):
    if self.mesh and 'xla' in str(self.fc2.weight.device):
      xs.mark_sharding(
          self.fc2.weight, self.mesh, (1, 0), use_dynamo_custom_op=True)
    y = self.relu(self.fc1(x))
    z = self.fc2(y)
    return self.fc3(z)


class DynamoSpmdInferenceTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()
    super().setUpClass()

  def test_dynamo_spmd_basic(self):
    device = xm.xla_device()
    linear = SimpleLinear().to(device)
    linear.eval()
    xla_x = torch.randn(1, 128, device=device)
    xs.mark_sharding(linear.fc2.weight, self._get_mesh((1, self.n_devices)),
                     (1, 0))
    xla_res = linear(xla_x)
    xm.mark_step()

    dynamo_linear = torch.compile(linear, backend="openxla")
    dynamo_res = dynamo_linear(xla_x)
    torch.allclose(xla_res.cpu(), dynamo_res.cpu())
    # TODO(JackCaoG): add counter checks after ExecuteReplicated also creates
    # a ExecuteMetric.

  def test_dynamo_spmd_output_sharding_spec(self):
    device = xm.xla_device()
    linear = SimpleLinear().to(device)
    linear.eval()
    xla_x = torch.randn(1, 128, device=device)
    xs.mark_sharding(linear.fc2.weight, self._get_mesh((1, self.n_devices)),
                     (1, 0))
    dynamo_linear = torch.compile(linear, backend="openxla")
    dynamo_res = dynamo_linear(xla_x)
    self.assertNotIn('ShardingSpec: None',
                     torch_xla._XLAC._get_xla_tensor_debug_info(dynamo_res))

  @unittest.skip(
      "test is flaky, UncachedOutputSharding sometime doesn't show up. most likely a waitdeviceop issue"
  )
  def test_dynamo_spmd_output_sharding_cache(self):
    met.clear_all()
    device = xm.xla_device()
    linear = SimpleLinear().to(device)
    linear.eval()
    xla_x = torch.randn(1, 128, device=device)
    xla_y = torch.randn(1, 128, device=device)
    xs.mark_sharding(linear.fc2.weight, self._get_mesh((1, self.n_devices)),
                     (1, 0))
    dynamo_linear = torch.compile(linear, backend="openxla")
    dynamo_res = dynamo_linear(xla_x)
    xm.wait_device_ops()
    self.assertIn('UncachedOutputSharding', met.counter_names())
    self.assertEqual(met.counter_value('UncachedOutputSharding'), 1)
    dynamo_res = dynamo_linear(xla_y)
    self.assertEqual(met.counter_value('UncachedOutputSharding'), 1)

  def test_dynamo_sharded_input(self):
    device = xm.xla_device()
    linear = SimpleLinear().to(device)
    linear.eval()
    xla_x = torch.randn(8, 128, device=device)
    xs.mark_sharding(xla_x, self._get_mesh((1, self.n_devices)), (1, 0))
    xla_res = linear(xla_x)
    xm.mark_step()

    dynamo_linear = torch.compile(linear, backend="openxla")
    dynamo_res = dynamo_linear(xla_x)
    torch.allclose(xla_res.cpu(), dynamo_res.cpu())

  def test_dynamo_input_sharding_changed(self):
    device = xm.xla_device()
    linear = SimpleLinear().to(device)
    linear.eval()
    xla_x = torch.randn(8, 128, device=device)
    xla_y = torch.randn(8, 128, device=device)
    xm.mark_step()

    met.clear_all()
    dynamo_linear = torch.compile(linear, backend="openxla")
    dynamo_res = dynamo_linear(xla_x)
    self.assertIn('CompileTime', met.metric_names())
    self.assertEqual(met.metric_data('CompileTime')[0], 1)

    # Shard the original input
    xs.mark_sharding(xla_x, self._get_mesh((1, self.n_devices)), (1, 0))
    dynamo_res_sharded = dynamo_linear(xla_x)
    torch.allclose(dynamo_res.cpu(), dynamo_res_sharded.cpu())
    # one graph is being generated by .cpu call above
    if self.n_devices > 1:
      self.assertEqual(met.metric_data('CompileTime')[0], 3)
    else:
      # if there is only one device(cpu) then sharding spec will be replicated
      # hence no change.
      self.assertEqual(met.metric_data('CompileTime')[0], 1)

    # Call the dynamo function with a different input with different sharding
    xs.mark_sharding(xla_y, self._get_mesh((1, self.n_devices)), (0, 1))
    dynamo_res_sharded_2 = dynamo_linear(xla_y)
    if self.n_devices > 1:
      self.assertEqual(met.metric_data('CompileTime')[0], 4)
    else:
      # if there is only one device(cpu) then sharding spec will be replicated
      # hence no change.
      self.assertEqual(met.metric_data('CompileTime')[0], 1)
    torch.allclose(linear(xla_y).cpu(), dynamo_res_sharded_2.cpu())

  @unittest.skipIf(xr.global_runtime_device_count() == 1,
                   "Multiple devices needed to test the mesh change")
  def test_dynamo_input_sharding_threashold(self):
    device = xm.xla_device()
    linear = SimpleLinear().to(device)
    linear.eval()
    xla_x = torch.randn(8, 128, device=device)
    xm.mark_step()

    dynamo_linear = torch.compile(linear, backend="openxla")
    if 'XLA_DYNAMO_INPUT_SHARDING_CHECK_THRESHOLD' in os.environ:
      saved_var = os.environ['XLA_DYNAMO_INPUT_SHARDING_CHECK_THRESHOLD']
    else:
      saved_var = None
    os.environ['XLA_DYNAMO_INPUT_SHARDING_CHECK_THRESHOLD'] = '2'

    # Execute the graph, threshold becomes 1 after this.
    dynamo_res = dynamo_linear(xla_x)
    # Change the sharding, given it is within threshold, following execution should
    # go through. Since the sharding change is detected, threshold will get reset to 2.
    xs.mark_sharding(xla_x, self._get_mesh((1, self.n_devices)), (1, 0))
    dynamo_res = dynamo_linear(xla_x)
    dynamo_res = dynamo_linear(xla_x)
    dynamo_res = dynamo_linear(xla_x)
    # Change the sharding again, now it exceed the threshold, we expect the graph execution
    # to fail.
    xs.clear_sharding(xla_x)
    xs.mark_sharding(xla_x, self._get_mesh((1, self.n_devices)), (0, 1))
    # crash will hapeen in a async execution thread, need to grab the lock again to
    # surface that exception
    dynamo_res = dynamo_linear(xla_x)
    try:
      print(dynamo_res)
    except:
      print('catch')
    # it is hard to catch the C++ runtime error in python, instead we can check if
    # after printing that dynamo_res is still a placeholder then it means C++ crashed.
    self.assertTrue(torch_xla._XLAC._is_placecholder(dynamo_res))
    if saved_var != None:
      os.environ['XLA_DYNAMO_INPUT_SHARDING_CHECK_THRESHOLD'] = saved_var
    else:
      del os.environ['XLA_DYNAMO_INPUT_SHARDING_CHECK_THRESHOLD']

  def test_dynamo_spmd_mark_sharding_outside_of_compile(self):
    device = xm.xla_device()
    linear = SimpleLinear().to(device)
    linear.eval()
    xla_x = torch.randn(1, 128, device=device)
    xs.mark_sharding(
        linear.fc2.weight,
        self._get_mesh((1, self.n_devices)), (1, 0),
        use_dynamo_custom_op=True)
    xla_res = linear(xla_x)
    xm.mark_step()

    dynamo_linear = torch.compile(linear, backend="openxla")
    dynamo_res = dynamo_linear(xla_x)
    torch.allclose(xla_res.cpu(), dynamo_res.cpu())

    # Ensure that another run with same input does not trigger additional compilation
    compile_count = met.metric_data('CompileTime')[0]
    dynamo_res = dynamo_linear(xla_x)
    self.assertEqual(met.metric_data('CompileTime')[0], compile_count)

  def test_mark_sharding_inside_compile(self):
    met.clear_counters()
    device = xm.xla_device()
    mesh = self._get_mesh((1, self.n_devices))

    # Passing this `mesh` as a parameter to `SimpleLinear` will call the dynamo custom op
    # variant of mark_sharding inside the forward function.
    linear = SimpleLinear(mesh=mesh).to(device)
    linear.eval()

    xla_x = torch.randn(1, 128, device=device)
    xla_res = linear(xla_x)
    xm.mark_step()

    dynamo_linear = torch.compile(linear, backend="openxla")
    dynamo_res = dynamo_linear(xla_x)
    torch.allclose(xla_res.cpu(), dynamo_res.cpu())

    # Ensure that another run with same input does not trigger additional compilation
    compile_count = met.metric_data('CompileTime')[0]
    dynamo_res = dynamo_linear(xla_x)
    self.assertEqual(met.metric_data('CompileTime')[0], compile_count)

  def test_dynamo_spmd_basic_with_dynamo_mark_sharding(self):
    device = xm.xla_device()
    linear = SimpleLinear().to(device)
    linear.eval()
    xla_x = torch.randn(1, 128, device=device)
    mesh = self._get_mesh((1, self.n_devices))
    device_ids = mesh.device_ids.tolist()
    mesh_shape = list(mesh.mesh_shape)
    axis_names = str(mesh.axis_names)
    partition_spec = '(1, 0)'
    torch.ops.xla.dynamo_mark_sharding(linear.fc2.weight, device_ids,
                                       mesh_shape, axis_names, partition_spec)
    xla_res = linear(xla_x)
    xm.mark_step()

    dynamo_linear = torch.compile(linear, backend="openxla")
    dynamo_res = dynamo_linear(xla_x)
    torch.allclose(xla_res.cpu(), dynamo_res.cpu())

  def test_dynamo_spmd_activation_sharding_with_dynamo_mark_sharding(self):
    def fn(t):
      t2 = t + 1
      device_ids = [0]
      mesh_shape = [1, 1]
      axis_names = 'None'
      partition_spec = '(1, 0)'
      torch.ops.xla.dynamo_mark_sharding(t2, device_ids, mesh_shape, axis_names, partition_spec)
      return t

    device = xm.xla_device()
    xla_x = torch.randn(1, 128, device=device)
    xla_res = fn(xla_x)
    xm.mark_step()

    dynamo_fn = torch.compile(fn, backend="openxla")
    dynamo_res = dynamo_fn(xla_x)
    torch.allclose(xla_res.cpu(), dynamo_res.cpu())


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
