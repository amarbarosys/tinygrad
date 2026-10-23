# Live check: generated kfd bindings (ioctl code + struct layout) drive a real AMDKFD_IOC_GET_VERSION.
from std.ffi import external_call
from kfd import AMDKFD_IOC_GET_VERSION, struct_kfd_ioctl_get_version_args

def main() raises:
    var path = String("/dev/kfd")
    var fd = external_call["open64", Int32](path.as_c_string_slice().unsafe_ptr(), Int32(2))
    if fd < 0:
        raise Error("FAIL open: /dev/kfd")
    var args = struct_kfd_ioctl_get_version_args()
    var rc = external_call["ioctl", Int32](fd, UInt64(AMDKFD_IOC_GET_VERSION), args._addr())
    if rc != 0:
        raise Error("FAIL ioctl: rc " + String(rc))
    print(args.major_version(), args.minor_version())
    _ = external_call["close", Int32](fd)
