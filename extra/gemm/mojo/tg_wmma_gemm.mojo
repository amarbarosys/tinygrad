# tinygrad-callable entry: three plain pointers (tinygrad's buffer ABI, no hidden kernargs) around the pipelined fp16 WMMA GEMM
# C[N,N] (fp32) = A[N,N] @ B[N,N] (fp16, row-major); N is fixed at build time (-D TG_N=4096), a multiple of 128
from max.gpu.host import DeviceContext
from std.sys import get_defined_int
from layout import TileTensor, row_major
from matmul_wmma_pipe import amar_matmul_wmma_pipe, C_DTYPE

comptime N = get_defined_int["TG_N", 4096]()
comptime a_layout = row_major[N, N]()
comptime b_layout = row_major[N, N]()
comptime c_layout = row_major[N, N]()

def tg_wmma_gemm(a: UnsafePointer[Float16, MutAnyOrigin], b: UnsafePointer[Float16, MutAnyOrigin], c: UnsafePointer[Scalar[C_DTYPE], MutAnyOrigin]):
    amar_matmul_wmma_pipe[type_of(a_layout), type_of(b_layout), type_of(c_layout)](
        TileTensor(a, a_layout), TileTensor(b, b_layout), TileTensor(c, c_layout), Int32(N), Int32(N), Int32(N))

def main() raises:
    var ctx = DeviceContext()
    _ = ctx.compile_function[tg_wmma_gemm]()
