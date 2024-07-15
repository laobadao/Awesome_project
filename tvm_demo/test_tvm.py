
import tvm
from tvm import te
import numpy as np

# 1. 定义矩阵乘法的大小
N = 512
A = te.placeholder((N, N), name='A')
B = te.placeholder((N, N), name='B')

# 2. 定义矩阵乘法计算
k = te.reduce_axis((0, N), 'k')
C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

# 3. 创建调度
s = te.create_schedule(C.op)

# 4. 编译计算图
func = tvm.build(s, [A, B, C], target='llvm')

# 5. 创建输入数据
a_np = np.random.uniform(size=(N, N)).astype(np.float32)
b_np = np.random.uniform(size=(N, N)).astype(np.float32)
c_np = np.zeros((N, N), dtype=np.float32)

# 6. 将数据传递到设备
a_tvm = tvm.nd.array(a_np)
b_tvm = tvm.nd.array(b_np)
c_tvm = tvm.nd.array(c_np)

# 7. 执行计算
func(a_tvm, b_tvm, c_tvm)

# 8. 验证结果
np.testing.assert_allclose(c_tvm.asnumpy(), np.dot(a_np, b_np), rtol=1e-5)
print("矩阵乘法计算正确")

# 9. 输出结果
print(c_tvm)
