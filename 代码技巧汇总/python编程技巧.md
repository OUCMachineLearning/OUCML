# Python 3 新特性：类型注解

前几天有同学问到，这个写法是什么意思：

```python3
def add(x:int, y:int) -> int:
    return x + y
```

我们知道 Python 是一种动态语言，变量以及函数的参数是**不区分类型**。因此我们定义函数只需要这样写就可以了：

```python3
def add(x, y):
    return x + y
```

这样的好处是有极大的灵活性，但坏处就是对于别人代码，无法一眼判断出参数的类型，IDE 也无法给出正确的提示。

于是 Python 3 提供了一个新的特性：
**函数注解**

也就是文章开头的这个例子：

```python3
def add(x:int, y:int) -> int:
    return x + y
```

用 `: 类型` 的形式指定函数的**参数类型**，用 `-> 类型` 的形式指定函数的**返回值**类型。

然后特别要强调的是，Python 解释器**并不会**因为这些注解而提供额外的校验，没有任何的类型检查工作。也就是说，这些类型注解加不加，对你的代码来说**没有任何影响**：

![img](https://pic4.zhimg.com/80/v2-9b87d43afdc941929c428b03865037c3_hd.jpg)

输出：

![img](https://pic4.zhimg.com/80/v2-f00b176bddec7e1d242e3c0e13f30f33_hd.jpg)

但这么做的好处是：

1. 让别的程序员看得更明白
2. 让 IDE 了解类型，从而提供更准确的代码提示、补全和语法检查（包括类型检查，可以看到 str 和 float 类型的参数被高亮提示）

![img](https://pic2.zhimg.com/80/v2-057629e3d59552c856dba7341882ea55_hd.jpg)

在函数的 `__annotations__` 属性中会有你设定的注解：

![img](https://pic1.zhimg.com/80/v2-9a9f55559930501472752996a3b772a4_hd.jpg)

输出：

![img](https://pic4.zhimg.com/80/v2-35506981e2c4f0e021b4b0902e36441b_hd.jpg)

在 Python 3.6 中，又引入了对**变量类型**进行注解的方法：

```python3
a: int = 123
b: str = 'hello'
```

更进一步，如果你需要指明一个全部由整数组成的列表：

```python3
from typing import List
l: List[int] = [1, 2, 3]
```

但同样，这些仅仅是“**注解**”，不会对代码产生任何影响。

不过，你可以通过 **mypy** 库来检验最终代码是否符合注解。

安装 mypy：

```bash
pip install mypy
```

执行代码：

```bash
mypy test.py
```

如果类型都符合，则不会有任何输出，否则就会给出类似输出：

![img](https://pic4.zhimg.com/80/v2-57b0ba00b7e95b092ae6fd6acc50eb37_hd.jpg)

这些新特性也许你并不会在代码中使用，不过当你在别人的代码中看到时，请按照对方的约定进行赋值或调用。

当然，也不排除 Python 以后的版本把类型检查做到解释器里，谁知道呢。