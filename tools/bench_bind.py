# Python twin of mojo/test/bench_bind.mojo: same ops through tinygrad's ctypes bindings.
import os, time
from tinygrad.runtime.autogen import kfd
s, n = kfd.struct_kfd_ioctl_update_queue_args(), 1_000_000
for _ in range(3):
  t, acc = time.perf_counter_ns(), 0
  for i in range(n):
    s.ring_base_address = i
    acc += s.ring_base_address + s.queue_id
  print("field ns/op", (time.perf_counter_ns() - t) / n)
fd, m = os.open("/dev/kfd", os.O_RDWR), 200_000
for _ in range(3):
  t = time.perf_counter_ns()
  for _ in range(m): a = kfd.AMDKFD_IOC_GET_VERSION(fd)
  print("ioctl ns/op", (time.perf_counter_ns() - t) / m, a.minor_version)
