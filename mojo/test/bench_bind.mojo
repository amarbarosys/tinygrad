# Cost per op of generated bindings: struct field set+get, and a real ioctl (pairs with tools/bench_bind.py).
from std.ffi import external_call
from std.time import perf_counter_ns
from std.benchmark import keep
from kfd import AMDKFD_IOC_GET_VERSION, struct_kfd_ioctl_update_queue_args, struct_kfd_ioctl_get_version_args

def main() raises:
    var s = struct_kfd_ioctl_update_queue_args()
    var n = 10_000_000
    var acc: UInt64 = 0
    for _ in range(3):
        var t = perf_counter_ns()
        for i in range(n):
            s.set_ring_base_address(UInt64(i))
            acc += s.ring_base_address() + UInt64(s.queue_id())
            keep(s._addr())
        print("field ns/op", Float64(perf_counter_ns() - t) / Float64(n))
    keep(acc)
    var path = String("/dev/kfd")
    var fd = external_call["open64", Int32](path.as_c_string_slice().unsafe_ptr(), Int32(2))
    var a = struct_kfd_ioctl_get_version_args()
    var m = 200_000
    for _ in range(3):
        var t = perf_counter_ns()
        for _ in range(m):
            if external_call["ioctl", Int32](fd, UInt64(AMDKFD_IOC_GET_VERSION), a._addr()) != 0:
                raise Error("FAIL ioctl")
        print("ioctl ns/op", Float64(perf_counter_ns() - t) / Float64(m), a.minor_version())
