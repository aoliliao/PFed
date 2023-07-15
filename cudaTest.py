import torch
import random

from util.result_util import set_fixed_seed
from util.utils import generate_projection_matrix, unit_test_projection_matrices



def run():
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        print("CUDA is available")
    else:
        device = torch.device("cpu")
        print("CUDA is not available")

    # 创建一个简单的模型
    model = torch.nn.Linear(10, 1).to(device)

    # 打印模型的随机权重
    print("Initial model weights:")
    print(model.weight)

    # 随机输入数据
    input_data = torch.randn(1, 10).to(device)

    # 在CUDA上进行前向传播
    output = model(input_data)

    # 打印前向传播结果
    print("Output:")
    print(output)

    # 检查是否固定了随机结果
    print("Are the results reproducible?")
    reproducible = torch.all(torch.eq(output, torch.tensor([[0.0446]], device=device)))
    print(reproducible)

if __name__ == '__main__':
    set_fixed_seed()
    projections = generate_projection_matrix(num_client=4, feature_dim=512, qr=False)
    unit_test_projection_matrices(projections)
    run()
