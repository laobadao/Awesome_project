import torch
import torchvision.models as models
import tvm
from tvm import relay
from tvm.contrib import graph_executor

# 1. 加载预训练的 PyTorch 模型
model = models.resnet18(pretrained=True)
model.eval()

# 2. 定义输入数据的形状
input_shape = (1, 3, 224, 224)
input_data = torch.randn(input_shape)

# 3. 转换 PyTorch 模型为 Relay 计算图
scripted_model = torch.jit.trace(model, input_data).eval()
shape_list = [("input0", input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# 4. 设置目标硬件和编译选项
target = "llvm"  # 使用 CPU 作为目标硬件
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

# 5. 创建 TVM 运行时模块
dev = tvm.cpu(0)
module = graph_executor.GraphModule(lib["default"](dev))

# 6. 将输入数据传递给 TVM 模块
tvm_input = tvm.nd.array(input_data.numpy())
module.set_input("input0", tvm_input)

# 7. 运行模型并获取输出
module.run()
output = module.get_output(0).asnumpy()

# 8. 打印输出形状
print("Output shape:", output.shape)
