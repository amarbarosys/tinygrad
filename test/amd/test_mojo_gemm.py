# fork: Tensor.dot routes fp16 matmuls of 128-multiple shapes on gfx1100 to the Mojo WMMA kernel (tinygrad/runtime/support/mojo_gemm.py)
# needs the real card and `mojo` (or a filled cache): PYTHONPATH=. DEV=AMD python -m pytest test/amd/test_mojo_gemm.py
import os, shutil, unittest
import numpy as np
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.runtime.support import mojo_gemm

def _can_run() -> bool:
  try: return Device.DEFAULT == "AMD" and Device["AMD"].arch == "gfx1100" and not type(Device["AMD"].iface).__name__.startswith("MOCK") \
    and bool(os.environ.get("MOJO") or shutil.which("mojo"))
  except Exception: return False

@unittest.skipUnless(_can_run(), "needs a gfx1100 AMD device and mojo")
class TestMojoGemm(unittest.TestCase):
  def setUp(self):
    self.ctx = Context(MOJO_GEMM=1)
    self.ctx.__enter__()
  def tearDown(self): self.ctx.__exit__(None, None, None)

  def test_off_by_default(self):
    with Context(MOJO_GEMM=0): self.assertIsNone(mojo_gemm.dot(Tensor.randn(128, 128, dtype=dtypes.half), Tensor.randn(128, 128, dtype=dtypes.half)))

  def _check(self, xs, ws, transpose_w=False):
    x = Tensor.randn(*xs, dtype=dtypes.half).realize()
    w = Tensor.randn(*ws, dtype=dtypes.half).realize()
    wt = w.T if transpose_w else w
    self.assertIsNotNone(mojo_gemm.dot(x, wt), "eligible shape was not routed to the Mojo kernel")
    out = (x @ wt).realize()
    ref = (x.float() @ wt.float()).realize()
    self.assertEqual(out.shape, ref.shape)
    out_np, ref_np = out.numpy().astype(np.float32), ref.numpy()
    self.assertLess((np.square(out_np - ref_np)).mean(), 1e-2)
    self.assertLess(np.abs(out_np - ref_np).max() / np.abs(ref_np).max(), 2e-3)

  def test_square(self): self._check((1024, 1024), (1024, 1024))
  def test_small_tiles(self): self._check((512, 2048), (2048, 512))
  def test_linear_batched(self): self._check((2, 256, 2048), (8192, 2048), transpose_w=True)
  def test_linear_small_tiles(self): self._check((512, 8192), (2048, 8192), transpose_w=True)

  def test_shapes_sweep(self):
    shapes = [
      (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192),
      (2048, 4096, 4096), (2048, 4096, 14336), (2048, 14336, 4096), (2048, 4096, 1024),
      (1024, 2048, 4096),  # non-square M < N
      (4096, 2048, 1024),  # non-square M > N
    ]
    for M, K, N in shapes:
      for transpose_w in (False, True):
        with self.subTest(M=M, K=K, N=N, transpose_w=transpose_w):
          ws = (N, K) if transpose_w else (K, N)
          self._check((M, K), ws, transpose_w=transpose_w)

  def test_falls_back(self):
    x, w = Tensor.randn(100, 128, dtype=dtypes.half), Tensor.randn(128, 128, dtype=dtypes.half)
    self.assertIsNone(mojo_gemm.dot(x, w))
    self.assertIsNone(mojo_gemm.dot(Tensor.randn(128, 128), Tensor.randn(128, 128)))  # fp32
    np.testing.assert_allclose((x @ w).numpy(), x.numpy().astype(np.float32) @ w.numpy().astype(np.float32), rtol=2e-2, atol=2e-1)

  def test_backward(self):
    for transpose_w in (False, True):
      x = Tensor.randn(256, 512, dtype=dtypes.half).realize()
      w = Tensor.randn(*((384, 512) if transpose_w else (512, 384)), dtype=dtypes.half).realize()
      wt = w.T if transpose_w else w
      self.assertIsNotNone(mojo_gemm.dot(x, wt))
      (x @ wt).float().sum().backward()
      g = np.ones((256, 384), np.float32)
      xn, wtn = x.numpy().astype(np.float32), wt.numpy().astype(np.float32)
      np.testing.assert_allclose(x.grad.numpy().astype(np.float32), g @ wtn.T, rtol=1e-2, atol=1e-1)
      want_w = xn.T @ g
      np.testing.assert_allclose(w.grad.numpy().astype(np.float32), want_w.T if transpose_w else want_w, rtol=1e-2, atol=1e-1)

if __name__ == "__main__":
  unittest.main()
