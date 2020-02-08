# 参考资料：
# [1] https://mp.weixin.qq.com/s/MxvA-f6ocpmGLW5JQ8qsSg
# [2] http://cs231n.github.io/python-numpy-tutorial/

# Numpy（Numerical Python的简称）是高性能科学计算和数据分析的基础包，其提供了矩阵运算的功能。
# 除了科学计算用途之外，Numpy还可以用作通用数据的高效多维容器，定义任意的数据类型。这些都使得Numpy能够无缝、快速地与各种数据库集成。
# 在学习图像识别的过程中，需要将图片转换为矩阵。即将对图片的处理简化为向量空间中的向量运算。基于向量运算，我们就可以实现图像的识别。
#
# Numpy提供的主要功能具体如下：
#     ndarray——一个具有向量算术运算和复杂广播能力的多维数组对象。
#     用于对数组数据进行快速运算的标准数学函数。
#     用于读写磁盘数据的工具以及用于操作内存映射文件的工具。
#     非常有用的线性代数，傅里叶变换和随机数操作。
#     用于集成C /C++和Fortran代码的工具。


# Python有两种运行方式：交互式和脚本式
# 如果想查看下列代码的运行结果，可以选中多行，以交互式方式运行。
# 先运行下面一行代码导入包numpy（只需导入一次），然后选中需要运行的代码行。（vscode中选中多行的方式为 Alt+左键）

import numpy as np

# 01 创建数组
# 借用线性代数的说法，一维数组通常称为向量（vector），二维数组通常称为矩阵（matrix）

# Numpy中的array()方法创建向量
vector = np.array([1, 2, 3, 4])
print(vector)

# numpy.array()方法创建矩阵
matrix = np.array([[1, 'Tim'], [2, 'Joey'], [3, 'Johnny'], [4, 'Frank']])
print(matrix)


# 02 创建Numpy数组
# 通过Python列表解析的方式来创建Numpy数组
array = np.array([i for i in range(10)])
print(array)
# 也可以通过Python列表的方式来修改值
array[0] = 10
print(array)

# Numpy数组封装了其他方法来创建特殊矩阵
# numpy.zeros()方法用来创建数值都为0的数组
# 函数原型：zeros(shape, dtype=float, order='C')
a = np.zeros(10)
print(a)
print(a.dtype)  # 默认为类型为 float64
# 在创建Numpy数组的时候可以对dtype参数进行指定
a = np.zeros(10, dtype=int)
print(a)
print(a.dtype)
# 参数size为元组时，创建多维全0矩阵
a = np.zeros((3, 4))  # 创建一个3行4列的矩阵且其数据类型为float64
print(a)
print(a.dtype)

# np.ones()方法创建的数组的数值都为1，使用方式与np.zeros()方法相似
a = np.ones((3, 4))
print(a)

# np.full()方法可以创建指定值的数组
a = np.full((3, 5), 121)
print(a)

# np.eye()方法创建单位矩阵
eye = np.eye(2)
print(eye)

# 可以使用np.arange方法来创建Numpy的数组
# arange(start=None, stop, step=None, , dtype=None)
# 与Python中的range方法相似，arange也是前闭后开的方法
# start指定开始的值，默认为0
# stop指定结束的值
# step指定步长，默认为1
a = np.arange(0, 20, 2)
print(a)

# 可以使用np.linspace方法（前闭后闭）生成等分矩阵
# 将0～10五等分的代码如下：
a = np.linspace(0, 10, 5)
print(a)

# np.random模块用于生成随机数组
# np.random.randint()生成随机整数数组
# 函数原型：randint(low, high=None, size=None, dtype='l')
a = np.random.randint(0, 5, 10)  # 长度为10的向量
print(a)
a = np.random.randint(4, 9, (3, 5))   # 3行5列的矩阵
print(a)

# np.random.random()方法生成0~1之间的随机浮点数数组
# 函数原型random(size=None)
np.random.random(10)  # 生成0~1之间的浮点数，向量的长度为10
np.random.random((2, 4))  # 生成0~1之间的浮点数，二行四列的矩阵

# np.random.normal()方法生成符合正态分布的随机矩数组
# 函数原型：normal(loc=0.0, scale=1.0, size=None)
# 参数：loc(float)，正态分布的均值；scale(float)，正态分布的标准差
a = np.random.normal(size=(3, 5))
print(a)

# np.random.seed()方法可以保证生成的随机数具有可预测性，
# 1.如果使用相同的seed()值，则每次生成的随即数都相同；
# 2.如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
for i in range(0, 5):
    np.random.seed(1)
    a = np.random.normal(size=(3, 5))
    print(a)


# 03 获取Numpy属性
# reshape(row,column)方法可以构架一个多行多列的array对象
a = np.arange(15).reshape(3, 5)  # 3行5列
print(a)
print(a.shape)  # 形状
print(a.ndim)   # 维数
print(a.dtype)  # 元素数据类型


# 04 Numpy数组索引
# Numpy支持类似list的定位操作
matrix = np.array([[1, 2, 3], [20, 30, 40]])
# 由于数组可能是多维的，因此必须为数组的每个维度指定索引
print(matrix[0, 1])  # 打印0行1列的元素

# 05 切片
# Numpy支持类似list的切片操作
matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
# 由于数组可能是多维的，因此必须为数组的每个维度指定索引
print(matrix[:, 0:2])   # 每行的第0和第1列元素
print(matrix[1:3, :])   # 第1和第2行的每列元素
print(matrix[1:3, 0:2])  # 行的索引是1和2，列的索引是0和1
# 也可以将整数索引与切片索引混合使用，这样做将产生比原始数组低级的数组
print(matrix[:, 1])     # 由每行的第1列元素组成的向量
# 可以使用...进行索引
print(matrix[..., 1])
print(matrix[...])

# 06 Numpy中的矩阵运算
# 两个矩阵的基本运算必须具有相同的行数与列数
myones = np.ones([3, 3])
myeye = np.eye(3)
print(myones)
print(myeye)
# +-*/执行的运算为两矩阵对应位置的元素进行运算
print(myones+myeye)
print(myones-myeye)
print(myones/myeye)
print(myones*myeye)
# 矩阵之间的点乘需使用函数dot()，以下两种方法均可
print(myeye.dot(myones))
print(np.dot(myeye, myones))

# 矩阵的转置
# 将原来矩阵中的行变为列，列变为行
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.T)  # 属性T可以获取转置矩阵

# 矩阵的逆
# 矩阵求逆必须行数和列数相同
A = np.array([[0, 1], [2, 3]])
invA = np.linalg.inv(A)
print(invA)
print(A.dot(invA))  # 原矩阵与逆矩阵点乘结果为单位矩阵

# Numpy其他预置函数
# 对矩阵a中的每个元素取正弦，sin(x)
np.sin(a)
# 对矩阵a中的每个元素取余弦，cos(x)
np.cos(a)
# 对矩阵a中的每个元素取正切，tan(x)
np.tan(a)
# 对矩阵a中的每个元素开根号
np.sqrt(a)
# 对矩阵a中的每个元素取绝对值
np.abs(a)


# 07 数据类型转换
# Numpy ndarray数据类型可以通过参数dtype进行设定，而且还可以使用astype来转换类型，
vector = np.array(["1", "2", "3"])
print(vector.dtype)
vector = vector.astype(float)   # 注意：如果字符串中包含非数字类型，那么从string转换成float就会报错。
print(vector.dtype)


# 广播
# 广播可以这么理解：当两个维度不同的数组（array）运算的时候，可以将低维的数组复制成高维数组参与运算（因为Numpy运算的时候需要结构相同）
x = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])
v = np.array([1, 0, 1])
y = x + v
print(y)
