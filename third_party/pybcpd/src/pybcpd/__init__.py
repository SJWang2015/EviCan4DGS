from ._bcpd import bcpd

# 变换类型枚举
class TransformationType:
    TSRN = 0  # 全部变换（平移、缩放、旋转、非刚性）
    TAN = 1   # 仿射+非刚性
    TA = 2    # 仅仿射
    TSR = 3   # 相似性（平移、缩放、旋转）
    TR = 4    # 刚性（平移、旋转）
    TN = 5    # 仅非刚性

# 保存选项常量，对应于C代码中的位掩码
class SaveOptions:
    SAVE_X = 0x0001      # 保存x
    SAVE_Y = 0x0002      # 保存y
    SAVE_V = 0x0004      # 保存v
    SAVE_A = 0x0008      # 保存a
    SAVE_C = 0x0010      # 保存c
    SAVE_E = 0x0020      # 保存e
    SAVE_P = 0x0040      # 保存P
    SAVE_T = 0x0080      # 保存T
    SAVE_PATHX = 0x0100  # 保存X
    SAVE_PATHY = 0x0200  # 保存Y
    SAVE_PFLOG = 0x0400  # 保存t
    SAVE_VTIME = 0x0800  # 保存0
    SAVE_ALL = 0x1000    # 保存A
    
    # 常用组合
    SAVE_YP = 0x0240     # 保存Y和P (0x0200 | 0x0040)
    
def register(
    X, Y, 
    random_seed=0, 
    omega=0.0, 
    lambda_param=2.0, 
    beta=2.0, 
    gamma=1.0,
    kappa=0.0,
    max_iter=50,
    min_iter=30,
    tolerance=1e-4,
    quiet_mode=False,
    accel_mode=False,
    nystrom_J=300,
    nystrom_K=70,
    use_kdtree=True,
    gauss_kernel_type=0,
    tau=0.0,
    knn=0,
    radius=0.0,
    normalization_type='e',
    transformation_type=TransformationType.TSRN,
    search_scale=7.0,
    search_radius=0.15,
    sigma_threshold=0.2,
    downsampling_flag=False,
    dwn_X=0,
    dwn_Y=0,
    dwr_X=0.0,
    dwr_Y=0.0,
    save_options=0x0240 
):
    """
    执行BCPD点云配准算法
    
    参数:
        X (numpy.ndarray): 目标点云, 形状 (N, D)
        Y (numpy.ndarray): 源点云, 形状 (M, D)
        random_seed (int): 随机种子
        omega (float): 离群点比率 [0,1)
        lambda_param (float): 正则化权重
        beta (float): 运动一致性参数
        gamma (float): 噪声参数
        kappa (float): 更新步长
        max_iter (int): 最大迭代次数
        min_iter (int): 最小迭代次数
        tolerance (float): 收敛阈值
        quiet_mode (bool): 是否安静模式
        accel_mode (bool): 是否启用加速
        nystrom_J (int): Nyström加速点数
        nystrom_K (int): Nyström加速秩
        use_kdtree (bool): 是否使用KD树
        normalization_type (str): 归一化类型 ('e','x','y','n')
        transformation_type (int): 变换类型
    
    返回:
        dict: 包含以下键的字典:
            - 'iterations': 迭代次数
            - 'transformed_points': 变换后的点云
            - 'displacement_field': 位移场
            - 'scale': 缩放因子
            - 'rotation': 旋转矩阵
            - 'translation': 平移向量
            - 'sigma': 最终sigma值
    """
    # 确保输入是连续的C顺序数组
    import numpy as np
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    
    # 处理整数参数 - 确保它们是整数
    random_seed = int(random_seed)
    max_iter = int(max_iter)
    min_iter = int(min_iter)
    nystrom_J = int(nystrom_J)
    # print("In __init__.py, nystrom_J:", nystrom_J)
    nystrom_K = int(nystrom_K)
    gauss_kernel_type = int(gauss_kernel_type)
    knn = int(knn)
    transformation_type = int(transformation_type)
    dwn_X = int(dwn_X)
    dwn_Y = int(dwn_Y)
    save_options = int(save_options)
    
    # 处理布尔值参数 - 转换为整数
    quiet_mode = int(bool(quiet_mode))
    accel_mode = int(bool(accel_mode))
    use_kdtree = int(bool(use_kdtree))
    downsampling_flag = int(bool(downsampling_flag))
    
    # 处理normalization_type - 转换为整数ASCII码
    if isinstance(normalization_type, str) and len(normalization_type) > 0:
        normalization_type = ord(normalization_type[0])
    elif isinstance(normalization_type, (int, float)):
        normalization_type = int(normalization_type)
    else:
        normalization_type = ord('e')  # 默认值
    
    # 调用C API
    return bcpd(
        X, Y, 
        random_seed, 
        omega, 
        lambda_param,  # 避免与Python关键字lambda冲突
        beta, 
        gamma,
        kappa,
        max_iter,
        min_iter,
        tolerance,
        quiet_mode,
        accel_mode,
        nystrom_J,
        nystrom_K,
        use_kdtree,
        gauss_kernel_type,
        tau,
        knn,
        radius,
        normalization_type,  # 现在是ASCII码整数
        transformation_type,
        search_scale,
        search_radius,
        sigma_threshold,
        downsampling_flag,
        dwn_X,
        dwn_Y,
        dwr_X,
        dwr_Y,
        save_options
    )
  
                                            
   