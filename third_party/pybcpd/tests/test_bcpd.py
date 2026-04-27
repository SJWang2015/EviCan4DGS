import numpy as np
import argparse
import time
from pybcpd import bcpd, bcpd_advanced, TransformationType

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BCPD测试')
    
    # 添加命令行参数
    parser.add_argument('-w', '--omega', type=float, default=0.0, help='离群点概率')
    parser.add_argument('-b', '--beta', type=float, default=2.0, help='核函数beta参数')
    parser.add_argument('-l', '--lambda', dest='lambda_', type=float, default=20.0, help='正则化权重')
    parser.add_argument('-g', '--gamma', type=float, default=10.0, help='核函数gamma参数')
    parser.add_argument('-J', '--nystrom-J', type=int, default=300, help='Nyström方法参数J')
    parser.add_argument('-K', '--nystrom-K', type=int, default=70, help='Nyström方法参数K')
    parser.add_argument('-p', '--accelerate', action='store_true', help='使用加速BCPD')
    parser.add_argument('-c', '--tolerance', type=float, default=1e-6, help='收敛阈值')
    parser.add_argument('-n', '--max-iter', type=int, default=500, help='最大迭代次数')
    parser.add_argument('-H', '--history', action='store_true', help='显示收敛历史')
    parser.add_argument('-r', '--random-seed', type=int, default=1, help='Random seed')
    # parser.add_argument('-r', '--random-seed', type=int, default=1, help='使用刚性变换类型')
    parser.add_argument('-s', '--save', type=str, default='Y', help='保存变换后的点云')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print(f"BCPD测试运行参数:")
    print(f"  omega = {args.omega}")
    print(f"  beta = {args.beta}")
    print(f"  lambda = {args.lambda_}")
    print(f"  gamma = {args.gamma}")
    print(f"  nystrom_J = {args.nystrom_J}")
    print(f"  nystrom_K = {args.nystrom_K}")
    print(f"  accelerate = {args.accelerate}")
    print(f"  tolerance = {args.tolerance}")
    print(f"  max_iter = {args.max_iter}")
    print(f"  history = {args.history}")
    print(f"  random_seed = {args.random_seed}")
    # print(f"  rigid = {args.rigid}")
    print(f"  save = {args.save}")
    
    # 创建示例点云数据（这里替换为您自己的数据）
    print("生成测试数据...")
    N = 1000  # X点云的点数
    M = 800   # Y点云的点数
    D = 3     # 维度
    
    # 随机生成点云数据
    np.random.seed(42)  # 固定随机种子以便结果可复现
    X = np.random.randn(N, D)
    
    # 创建Y点云（添加旋转、平移和噪声）
    theta = np.pi / 6  # 30度旋转
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0], 
        [0, 0, 1]
    ])
    t = np.array([1.5, 0.8, -0.5])  # 平移向量
    s = 0.9  # 缩放因子
    
    # 对X的子集进行变换得到Y
    indices = np.random.choice(N, M, replace=False)
    Y = s * (X[indices] @ R.T) + t + 0.05 * np.random.randn(M, D)  # 添加噪声
    
    # 根据参数设置变换类型
    # if args.rigid == 1:
    #     transformation_type = TransformationType.TR  # 刚性变换（平移+旋转）
    # else:
    transformation_type = TransformationType.TSRN  # 默认：全变换
    
    # 运行BCPD算法
    print("运行BCPD算法...")
    start_time = time.time()
  
    # # 使用bcpd_advanced函数
    result = bcpd_advanced(
        X, Y, 
        random_seed=args.random_seed,
        omega=args.omega,
        lambda_=args.lambda_,
        beta=args.beta,
        gamma=args.gamma,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        accelerate=args.accelerate,
        nystrom_J=args.nystrom_J,
        nystrom_K=args.nystrom_K,
        transformation_type=transformation_type,
        return_dict=True
    )

    result = bcpd(X, Y,  
         omega=0.0, lambda_=20.0, gamma=10.0, beta=2.0, delta=7.0, 
         epsilon=1e-6, max_iterations=500, J=300, K=70, 
         options=0, return_dict=True)
    
    elapsed_time = time.time() - start_time
    
    # 输出结果
    print(f"计算完成! 耗时: {elapsed_time:.2f}秒")
    print(f"迭代次数: {result['iterations']}")
    print(f"最终sigma: {result['sigma']:.6f}")
    print(f"缩放系数: {result['scale']:.6f}")
    
    # 输出旋转矩阵和平移向量
    print("\n估计的旋转矩阵R:")
    print(result['rotation'])
    print("\n估计的平移向量t:")
    print(result['translation'])
    
    # 计算误差
    Y_transformed = result['Y_transformed']
    v_out = result['v_out']
    error = np.mean(np.sqrt(np.sum((Y_transformed - Y)**2, axis=1)))
    print(f"\n平均变换误差: {error:.6f}")
    
    # 保存结果
    if args.save == 'Y':
        np.savetxt('Y_transformed.txt', Y_transformed)
        print("变换后的Y点云已保存至 Y_transformed.txt")
        np.savetxt('v_out.txt', v_out)
        print("变换后的Y点云已保存至 v_out.txt")
    
    print("\nBCPD测试完成!")

if __name__ == '__main__':
    main()