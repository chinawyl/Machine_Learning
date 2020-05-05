#### 一、基本操作

```octave
#1.加 减 乘 除 乘方
octave:3> 5+6
ans =  11
octave:4> 3-2
ans =  1
octave:5> 8*9
ans =  72
octave:6> 8/4
ans =  2
octave:7> 6/4
ans =  1.5000
octave:8> 2^6
ans =  64
octave:9>

#2.逻辑运算
octave:9> 1 == 2 %false
ans = 0
octave:10> 1~=2
ans = 1
octave:11> 1 && 0 % AND
ans = 0
octave:12> 1 || 0 % OR
ans = 1
octave:13> xor(1,0) % 异或
ans = 1
octave:14>

ps1('>>'); %改变提示符

#3.变量
>> a=3
a =  3
>> a=3;%加上分号将不会显示
>> a
a =  3
>> a=3;
>> a=2
a =  2
>> a=2;%加上分号将不会显示
>> b='hi' 
b = hi 
>> c=(3>=1) 
c = 1
>> a=pi
a =  3.1416
>> disp(a)
3.1416
>> disp(sprintf('2 decimals: %0.2f',a)) % 类似于C语言的输出语句
decimals: 3.14

#4.矩阵和向量
>> A = [1 2; 3 4; 5 6]
A =

   1   2
   3   4
   5   6

>> A=[1 2;
3 4;
5 6]
A =

   1   2
   3   4
   5   6

>> V=[1 2 3]
V =

   1   2   3

>> V=[1;2;3]
V =

   1
   2
   3
   
>> v = 1:0.1:2
v =

 Columns 1 through 4:

    1.000000000000000    1.100000000000000    1.200000000000000    1.300000000000000

 Columns 5 through 8:

    1.400000000000000    1.500000000000000    1.600000000000000    1.700000000000000

 Columns 9 through 11:

    1.800000000000000    1.900000000000000    2.000000000000000

>> V=1:6
V =

   1   2   3   4   5   6

>> ones(2,3)
ans =

   1   1   1
   1   1   1

>> 2*ones(2,3)
ans =

   2   2   2
   2   2   2

>> C=[2 2 2;2 2 2]
C =

   2   2   2
   2   2   2

>> w=ones(1,3)
w =

   1   1   1

>> w=zeros(1,3)
w =

   0   0   0
   
>> w=rand(1,3)
w =

   5.953529564162222e-01   1.922508972220850e-01   8.377724087620767e-01

>> w=rand(3,3) %产生随机数矩阵，范围0~1
w =

   3.294908206519311e-01   9.617127507689670e-01   8.979591799043839e-01
   5.681428442751816e-01   9.035147872371009e-01   6.775797827827869e-01
   2.914739899534777e-01   9.294818874219493e-01   3.992575655946695e-01

>> w = randn(1,3) %产生服从高斯分量的随机矩阵
w =

   1.330090760993083e+00   5.960338781756184e-01  -3.233255733034180e-02

>> w=-6 + sqrt(10)*(randn(1,10000));
>> hist(w) %绘制直方图
>> hist(w,50) %绘制更多竖条的直方图（50个）

>> i=eye(4) %单位矩阵
i =

Diagonal Matrix

   1   0   0   0
   0   1   0   0
   0   0   1   0
   0   0   0   1
   
>>help eye %查看帮助
```

#### 二、移动数据

```octave
A = [1 2;3 4; 5 6]
A =

   1   2
   3   4
   5   6
   
   
#size()返回矩阵的大小

size(A)
ans =

   3   2
#返回A的行数
size(A,1)
ans =  3

#返回A的列数
size(A,2)
ans =  2

#返回v的最大维数

>> v = [1 2 3 4]
v =

   1   2   3   4

>> length(v)
ans =  4

#返回A的最大维数（A 3*2）

>> length(A)
ans =  3
 

 

#文件处理
#octave的安装路径
>> pwd
ans = G:\octave\octave-5.1.0-w64

#ls显示当前目录下的文件结构

#cd进入路径

#加载数据
load   featureX.dat

load('featureX.dat')

#who显示定义的所有变量
>> who
Variables in the current scope:

A    I    a    ans  c    sz   v    w

#whos显示更详细的信息
>> whos
Variables in the current scope:

   Attr Name        Size                     Bytes  Class
   ==== ====        ====                     =====  =====
        A           3x2                         48  double
        I           4x4                         32  double
        a           1x1                          8  double
        ans         1x25                        25  char
        c           3x4                         96  double
        sz          1x2                         16  double
        v           1x4                         32  double
        w           1x10000                  80000  double

Total is 10066 elements using 80257 bytes

 
#clear删除变量
>> clear I
>> who % I已经不见了
Variables in the current scope: 

A    a    ans  c    sz   v    w

#将priceY的前10个数据赋值给v
>> v = priceY(1:10)
v =

   0.134924
   0.065742
   0.863374
   0.139802
   0.677121
   0.654762
   0.141145
   0.228309
   0.789854
   0.451826

#将数据存储在硬盘里
>> save hello.txt v;
>> save hello.txt v -ascii; % save as text(ASCII)ASCII格式

#clear 删除所有的变量
>>clear

#A(m,n)获取矩阵中特定的值
>> A
A =

   1   2
   3   4
   5   6
  >> A(3,2)
  ans = 6

#A(m,:)获取第m行的数据

>> A(2,:)
ans =

   3   4

#A(:,n)获取第n列的数据
>> A(:,1) %：冒号代表该行或者该列所有的数据
ans =

   1
   3
   5

#显示A矩阵第1行和第3行的数据
>> A([1 3], :)
ans =

   1   2
   5   6

#按行列替换元素
>> A(:,2) = [10;11;12]
A =

    1   10
    3   11
    5   12

#添加列向量B = [A , [column vector]]
>> A = [A,[100; 101; 102]]
A =

     1    10   100
     3    11   101
     5    12   102

#将A作为列向量显示
>> A(:)
ans =

     1
     3
     5
    10
    11
    12
   100
   101
   102

#结合两个矩阵横向放置C = [A B]
>>  B = [A,A]
B =

     1    10   100     1    10   100
     3    11   101     3    11   101
     5    12   102     5    12   102

#结合两个矩阵竖向放置C = [A ; B]
>> B = [A;A]
B =

     1    10   100
     3    11   101
     5    12   102
     1    10   100
     3    11   101
     5    12   102
```

#### 三、计算数据

```octave
#初始化数据
>> A = [1 2; 3 4; 5 6]
A =

   1   2
   3   4
   5   6

>> B = [11 12; 13 14; 15 16]
B =

   11   12
   13   14
   15   16

>> C = [1 1; 2 2]
C =

   1   1
   2   2

#矩阵乘法 A * B
>> A * C
ans =

    5    5
   11   11
   17   17

#矩阵点乘 A .* B (A中每一个相应的元素乘以B中的每一个元素)
>> A.*B
ans =

   11   24
   39   56
   75   96

#对矩阵元素幂运算 A .^ m
>> A .^ 2
ans =

    1    4
    9   16
   25   36

#取所有元素的倒数 1 ./ A
>> 1 ./ A
ans =

   1.00000   0.50000
   0.33333   0.25000
   0.20000   0.16667

#对数运算log(A)
>> log(A)
ans =

   0.00000   0.69315
   1.09861   1.38629
   1.60944   1.79176

#exp(A) 以e为底 A中每个元素为幂的幂运算
>> exp(A)
ans =

     2.7183     7.3891
    20.0855    54.5982
   148.4132   403.4288

#abs(A) 取绝对值
>> abs([-1 -2 -3])
ans =

1 2 3

#A' (A的转置矩阵)
>> A'
ans =

   1   3   5
   2   4   6

#val = max(v)列向量最大值
[val, ind] = max(v) % val最大值， ind下标索引
>> [val, ind] = max(A)
val =

   5   6

ind =

   3   3

#A < n % 矩阵A中小于n的返回1， 大于等于n的返回0
>> A < 3
ans =

  1  1
  0  0
  0  0

#[r, c] = find(A < 3) % 返回小于3的元素的索引
>> [r, c] = find(A < 3)
r =

   1
   1

c =

   1
   2

sum(A) %求和

sum(A, 1) %每一列之和

sum(A, 2) %每一行之和

prod(A) %求乘积

floor(A) %向下取整

ceil(A) %向上取整

 

#max(A, [], 1)  %每列最大值
#max(A, [], 2) %每行最大值
>> max(A, [], 1)
ans =

   8   9   7

>> max(A, [], 2)
ans =

   8
   7
   9

#flipud(A) 对矩阵垂直翻转
>> flipud(eye(3))
ans =

Permutation Matrix

   0   0   1
   0   1   0
   1   0   0

#pinv(A) 求逆矩阵
>> pinv(A)
ans =

   0.147222  -0.144444   0.063889
  -0.061111   0.022222   0.105556
  -0.019444   0.188889  -0.102778
```

### 四、数据绘制

```octave
#绘制折线
>> t = [0:0.01:0.98];
>> y1 = sin(2*pi*4*t);
>> plot(t, y1);

#hold on 在原来的图片上继续绘制
>> y2 = cos(2*pi*4*t);
>> hold on;
>> plot(t, y2, 'r'); % ‘r’ 红色

#添加标签
>> xlabel('time')
>> ylabel('value')
>> legend('sin', 'cos') % 标识数据
>> title('my plot') % 图片名称
>> print -dpng 'myplot.png' % 保存图片到当前目录

#将数据显示在不同的图片上
>> figure(1);plot(t, y1);
>> figure(2);plot(t, y2);

#分割图像subplot(a, b, c), a,b 将图像分割为a * b 的图像，c控制使用第几个图像
>> subplot(1,2,1)
>> plot(t,y1)
>> subplot(1,2,2)
>> plot(t,y2)
```

### 五、控制语句

```octave
#for循环
>> for i = 1 : 10,
v(i) = 2^i;
end;
>> v
v =

      2
      4
      8
     16
     32
     64
    128
    256
    512
   1024

>> indics = 1:10;
>> for i = indics,
disp(i);
end;
 1
 2
 3
 4
 5
 6
 7
 8
 9
 10

#while循环
>> i = 1;
>> while(i < 5),
v(i) = 10;
i++;
end;
>> v
v =

     10
     10
     10
     10
     32
     64
    128
    256
    512
   1024

#break
>> i = 1;
>> while true,
v(i) = 999;
i = i+1;
if i == 6,
  break;
end;
end;
>> v
v =

    999
    999
    999
    999
    999
     64
    128
    256
    512
   1024

#if 语句
>> if v(1) == 1,
       disp('The value is one');
   elseif v(1) == 2,
       disp('The value is two');
   else
       disp('The value is not one or two.');
   end;
The value is two

#函数定义 
#创建文件以      .m     结尾
function y = squareThisNumber(x)  % y是返回值

y = x^2;
>> squareThisNumber(5)
ans =  25

#添加搜索路径，让即使octave不在需要的路径下，也可以搜索到需要的文件
>> addpath('路径')

#函数返回多个值
#函数定义
　function [y1, y2] = squareAndCubeThisNumber(x)

　y1 = x^2;
　y2 = x^3;

#使用
>> [a, b] = squareAndCubeThisNumber(5)
a =  25
b =  125
```

