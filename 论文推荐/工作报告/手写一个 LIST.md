# 手写一个 LIST

面向对象(Object Orientied/OO)想必很多人对这个概念不陌生，但是不一定仔细了解过。事实上面向对象的范式能够大行其道和Java以及C++的受欢迎是分不开的。这两种语言比起更早期更基础的语言，针对面向对象编程进行了很多设计上的改进。而面向对象的范式在开发效率上，会比传统的工程模式更高，代码的迭代能力也会更强。面向对象是现在最主流的语言采用的最主流的编程范式，如果要写软件或应用类的代码，这种范式或者思想是必须掌握的。哪怕不去学习软件工程方面的技巧，不去学习编程范式，我觉得面向对象是必须要学的。

在正式开始前。我们再回忆一下介绍过的内容，学习变量类型是为了认识我们能够操作的对象，学习控制结构是为了掌握我们能够使用的基本操作方法。学习数据结构是为了帮助我们把操作对象进行一定的组合来改进操作的效率，学习函数是为了将操作方法进行整合。也就是说，**在Python基础中我们介绍了操作对象和操作方法，在Python进阶中我们介绍操作对象的组合和操作方法的整合**。那本文的面向对象自定义类，就是介绍除了简单数据结构和自定义函数以外，另一种既能组合操作对象又能整合操作方法的手段。

也就是说面向对象我暂时不作为编程范式来讲(篇幅所限)，我们只学习一下Python如何实现面向对象，他的实现有什么特点。我们先只涉及简单操作方法，具体的面向对象思想可能在后面的微型程序结构设计介绍一些编程范式的知识时介绍。

废话少说。我们直接尝试边学边实现一个之前在      讲过的"分离式动态顺序表"也就是Python内置的`list`。如果是零基础而且没看过本专题前面的文章，那么可能本文靠后的内容理解起来会具有一定的难度。另外提醒一下本文可能篇幅会比较长。

# Python的自定义类

说了这么多，面向对象到底是做什么的。这里的面向对象和我们之前一直强调的Python一切皆对象有什么关联。

我们已经无数次强调过Python的一切皆对象带来的一系列操作便利的地方。但是这两者的概念还是有一些区别的。我们之前涉及的都是Python内置的数据类型，是Python设计成对象方便使用的。而我们说面向对象编程，一般说的是自己去设计一个对象，这个对象会有一些特点，可以进行一些自定义的操作。

在Python中，我们通过自定义类`class`来实现这种对象设计。

```python
class Cat():
    def __init__(self):
        print('Meow')
```

这个就是一个最简单的类。可以看到和定义函数很类似，因为如果看作函数嵌套，那么上面的这段程序可以很容易改写成

```python
def cat():
    def meow():
        print('Meow')
    return meow()
```

它们用起来其实也很类似。

```python
>>> Cat()
Meow
<__main__.cat object at 0x00000207CB1F6710>
>>> cat()
Meow
```

唯一的区别我们看到自定义类似乎还返回了一个对象。这种东西一般函数作为对象时才会出现，比如

```python
>>> cat
<function cat_func at 0x00000207CA983E18>
```

和上面是不是很类似了。有人可能会猜测，是不是`Cat()`同时做了`cat()`和`cat`两种引用呢？所以才把两个结果都输出了。

如果你能这么想，那你就真的是很聪明了。事实上确实是这样。我们现在来看看Python的自定义类到底是应该长什么样子，以及刚才那个奇怪的`__init__`到底是什么东西。

我们讲过Python的自定义函数格式是这样的

```python
def func_name(arg1):
    pass
```

我刚才又说自定义类和函数很像，其实它们真的很像

```python
class ClsName(object):
    pass
```

这两段声明都是可以经过解释器检查有效的，我们注意到自定义类也可以带参数，这个参数到底是干什么的，我们后面再说。提醒一下Python中的类一般用大驼峰记名。

所以，刚才的`Cat()`和`cat()`真的就是嵌套函数把`def`改成了`class`。那到底有什么区别呢。

区别在于，自定义函数被调用以后，会引用它的返回值(没有就是`None`)，而直接引用这个函数，会引用这个函数对象，这个之前的文章分析过。对于自定义类来说，被调用以后会生成一个**类实例**并引用它，直接引用这个对象，会引用这个类对象。

```python
>>> some_val = cat()  # 调用，引用返回值
Meow
>>> type(some_val)  # 这里没有定义返回值，所以默认None
<class 'NoneType'>
>>> some_func = cat  # 引用，不会执行内部代码
>>> type(cat)
<class 'function'>
>>>
>>> # class below
>>> 
>>> Garfield = Cat()  # 调用，生成实例
Meow
>>> Garfield
<__main__.cat object at 0x00000207CB212668>
>>> type(Garfield)  # 这里可以看到是个Cat了
<class '__main__.Cat'>
>>> Exotic = Cat  # 引用，不会执行内部代码
>>> type(Exotic)
<class 'type'>
```

上面的对比可以清楚地看到，自定义函数是用`def`定义了一个`function`，而自定义类用`class`定义了一个`type`，我们平时用`type()`函数所要查看的，就是这个东西。

**总结起来**，自定义类其实和自定义函数非常类似。区别在于，它定义了一个新的**类型**，调用它的时候，就会产生一个这种类型的对象(叫作类实例)。就好像我定义了猫，那么我让某个变量`kitty=Cat()`，kitty就成了猫了。我们一般口语会说“kitty是一只猫”，但用编程的术语来说就变成了，“kitty变量是Cat类的一个实例”。总之，面向对象的技术让我们以后可以把所有不同名字的猫都变成猫类的实例，方便进行一些通用的操作和处理。

好了，上面是通过猫星人来介绍一下`class`大概是什么样子，它大概是如何工作的，下面会开始用不那么形象的例子来说明它到底具体是如何工作的，#我们来尝试手写一个Python列表list。

# 定义一个类

在刚才我们介绍Cat()的时候，讲到了调用以后会生成实例。但是这还是没有解决刚才的一个问题，就是刚才的`Meow`是如何输出的呢？

事实上，我们在函数内部定义函数叫嵌套函数，但是自定义类内部一般叫**方法(method)**。而且我们在外部是无法获取函数内部的嵌套函数的，只能通过闭包(closure)里的自引用，但是自定义类内部的变量和方法我们可以通过点方法来在全局引用。

```python
>>> class MyList(object):
...     author = 'sypy'
...     def __init__(self, data):
...         self.data = data
...
>>> MyList.author
'sypy'
>>> MyList.__init__
<function MyList.__init__ at 0x00000207CB1D7510>
```

可以看到，自定义类的方法确实是用`function`来实现，而且可以在外部进行引用的。如果你看了本专题前面的所有文章，那么我觉得你不难把对自定义类的方法的引用和局部变量的引用视作等价，它们其实都是一种**值访问**，具体该怎么理解我也不展开了，虽然这也是Python类的特点，但这不关键。我可能会在Python精进部分涉及到作用域和引用相关细节时会仔细分析一下这个问题。

而用双下划线包裹住的方法叫**魔术方法(magic method)**，是Python内置的一些对应特殊功能的方法，我们接下来会认识很多魔术方法，现在先说一下`__init__`，这个方法是初始化(initialize)方法，它特殊在会在生成类实例时自动调用。也就是说，刚才的类`Cat`实例化的时候自动调用了`__init__`函数，它输出了`Meow`。这也是为什么我们在自定义函数中把外层函数返回值设置成内部函数(或者在内部调用)才能调用内部的`meow()`函数来输出这个词。

所以说，刚才确实是`Cat()`类似于同时做了`cat()`和`cat`两种引用，一是生成了类实例，并且调用了初始化函数，二是引用了生成的类实例对象。

之所以一上来就介绍初始化函数，因为它是使用最广泛的几乎必备的类方法。

而`__init__`方法的第一个参数，以及类中绝大多数的普通方法(后面会介绍还有一些特殊方法)，都会有一个参数`self`。这个东西是什么呢？这个东西就是我们刚才说的**实例**，我们以后调用方法，一般是通过实例来调用，那么在这种情况下，类方法会把实例对象作为默认的第一个参数。也就是说`self`只是个名字，你也可以改成`ziji`，但是通用的是`self`(有的语言里面叫`this`)。这样我们可以在定义方法的函数体内通过`self.func`类似的点方法来调用实例的其他方法或者变量。

另一个值得一提的是我们这次定义类的时候加上了一个参数`object`。这个东西是Python的一个内置基础类

```python
>>> help(object)
Help on class object in module builtins:

class object
 |  The most base type
```

之所以会出现这种东西，也是历史遗留问题。在Python2中，**继承**自object的类会有更多的默认魔术方法，叫作**新式类**，而没有参数直接定义的类叫作**旧式类**(它们的继承追溯方法也不太一致，实例处理起来也有很多坑)。在Python3中已经没有这种说法了，已经统一用新式类，没有参数也默认继承自object，object这个字段也还在。我的建议是没必要偷这点懒，用IDE的话模板也会自动写上，也可以手动把object写上去，不要空在那里(理由和前面说字典最后一个元素最好也加上逗号一样，为的是格式统一)。

这里我们说到了一个新的概念，继承。

# 类的继承

我们刚刚介绍了如何定义一个类，发现和函数有一些区别，但是区别也不大，那为什么我们还需要类呢？

首先一个就是值访问的问题，这是类最大的特点，刚才已经说过了。但是类还有一个比函数更强大的地方，就是继承。

我们刚才自定义了一个类`Cat`，我说它是继承自`object`，而且`object`给了它许多默认的魔术方法。有很多我们无法直观地看出来，但是有一点是很明显的，我们为什么能够输出这个类的实例呢？

```python
>>> print(Garfield)
<__main__.cat object at 0x00000207CB212668>
```

我们理所当然地认为这个对象可以被输出，事实上不是这样的。它能够被输出，是因为object自带的`__repr__`方法被继承了。事实上它继承的方法非常多。

```python
>>> dir(Cat)
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__']
```

说了这么多，类的继承到底是什么呢？

类的继承其实就和我们之前讲过的函数装饰器差不多。我们想用某个函数完成一些额外的工作，又不想改变它的结构，那么我们可以用函数把它包裹起来，用参数把它传进装饰函数，在装饰函数里面调用它。

类的继承其实和函数装饰器意思差不多，但是要更强大。因为装饰器是能力有限的，我们无法访问函数内部的变量，无法直接调用它的内部函数。但是类不一样，我们继承一个类以后，可以用父类的方法，可以用父类的变量，可以写另外的方法完成更多功能，可以覆盖(override)原有的方法进行修改却不影响父类和其他继承自父类的子类。

```python
class MyMyList(MyList):
    pass
```

我们直接定义了一个继承自`MyList`的类，并且在内部没有作任何控制。看看它是否能和父类一样正常工作。

```python
>>> MyMyList.author
'sypy'
```

是不是比装饰器要简洁很多？我们没有必要特定去写一个装饰器该长什么样子，直接类继承就可以了。

而且，函数可以有多个参数，类一样可以有多个参数，这代表着同时继承自多个父类。这一般叫作**多继承**。

```python
class MyMyMyList(MyMyList, MyList):
    pass
```

甚至这样的父类本身相互间有继承关系也是允许的(当然并不建议这么做)。

```python
>>> MyMyMyList.author
'sypy'
>>> MyMyMyList.mro()
[<class '__main__.MyMyMyList'>, <class '__main__.MyMyList'>, <class '__main__.MyList'>, <class 'object'>]
```

上面的`mro`函数是用来查询类继承顺序的方法。

至于类继承关系复杂时如何追溯继承关系，如何处理冲突，我这里就不展开讲了。一方面我们没有介绍过算法，一方面难度不止Python进阶，一方面篇幅也有限。暂时我们可以尽量只使用单继承。这些内容预计也会在Python精进部分作用域和引用相关细节中介绍。

# 开始构造list

有了以上的基本知识，我们就可以开始构造自定义的list了。首先我们可以借用原生列表来简单试验一下我们刚才的知识。

```python
class MyList(object):
    def __init__(self, *args):
        self._data = []
        for arg in args:
            self._data.append(arg)
    
    def __repr__(self):
        return '%r' % self._data
```

这样我们就完成了一个最简单自定义列表。来试试看它能不能工作吧

```python
>>> test_list = MyList(1, 2, 4)
>>> test_list
[1, 2, 3]
>>> type(test_list)
<class '__main__.MyList'>
```

显然，几乎能以假乱真了。但是别高兴得太早，能输出是因为我重新覆盖了`__repr__`方法，所以能输出我想要的东西。其他列表的方法能不能实现呢

```python
>>> s.append(5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'MyList' object has no attribute 'append'
```

显然，一下子就露馅了。那么，我们的工作是不是开始一步步完成列表需要支持的方法呢？只要一个个写自定义函数，针对`self._val`进行一些操作就完了，剩下的都是体力活了。

不急。

既然要重新写，那为什么一定要依赖原生数据结构呢？为什么这么急着干体力活呢？但是，如何从无到有直接构造一个list结构出来呢？

在此之前我对刚才那个简单的实现方式还有一些细节需要说明。
 一个是`_data`，一个是`__repr__`。

#### 私有属性

首先“私有属性”这个名字是我自己起的，我也不知道中文别人管他叫什么，反正英文叫`private property`(或者`private method`，前面说过，它们本质上是一样的)，叫什么不关键，掌握了以后知道指什么东西就行了。

首先是`self._data`我们在定义类的属性的时候，用了下划线，在之前的文章中我们说这是出于安全考虑。一般来说，**前缀一个下划线的属性代表不希望被外部引用**，只在类内部工作。我特地说一个，是因为还有两个的，**前缀两个下划线的属性代表安全相关属性，一般是不允许被外部引用**。而后者，就是我要说的私有属性。

```python
>>> class t():
...     def __init__(self):
...         self.__val = 1
...
>>> ss = t()
>>> ss.__val
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 't' object has no attribute '__val'
```

很明显，我们在类内部设置了`self.__val = 1`，但是在外部是无法访问这个值的。你要说有没有办法访问，办法也是有的。

```python
>>> ss._t__val
1
```

看，出来了，前提是很麻烦。之所以这样这样设计，是为了对类属性进行区分。比如刚才提到的前缀一个下划线的变量，这种变量是不会被`from packagename import *`这样的方式引用的(虽然这种方式本身就是不建议使用的)。而形如`__val`这样的变量，一般是严格限制外部引用的。

#### repr和str

先说结论，一般来说，类都需要`__repr__`方法而不一定需要`__str__`方法。

我们使用命令行解释运行Python时，我们输出某个变量有时直接变量名回车就可以了，有时需要把它转成字符串用`print`函数输出。

```python
>>> s = 'Hello'
>>> s
'Hello'
>>> print(s)
Hello
```

可以看到，即使都可以用，有的时候输出结果也是不一样的。

这两个方法的区别在于，**`__repr__`是给机器或者程序员看的字符串，而`__str__`是给用户看的字符串**。一般遵循的原则是，`__str__`的结果就是用来给`print`函数输出的，而`__repr__`得到的字符串，有一部分甚至可以直接用`eval`函数让机器进一步执行(打基础的同学不建议用这种技巧)。

用`print`函数输出某个类实例时，会优先输出类`__str__`方法的返回值，如果没有就输出`__repr__`方法的返回值，如果没有自定义`__repr__`覆盖，那么原生的`object.__repr__`也直接输出该实例的类型信息和内存信息，比如刚才见过的`<__main__.cat object at 0x00000207CB212668>`。

所以，在设计类的时候，这两个函数的该完成的功能也是需要仔细考虑的。

好，回到刚才的问题，如何从0写一个列表呢。

我们首先考虑的是，列表该做什么，需要什么。在此基础上，我们给它什么就可以了。

列表需要一个容量来存储一些对象，需要索引来快速筛选对象。我们的列表要完成原生列表能完成的大部分功能。

这样，我们写出来的列表雏形大概是这样的

```python
class MyList(object):
    def __init__(self, data):
        self.__volume = 8
        self._data = data
```

没有了原生列表的支持，我们发现我们只能一次保存一个对象。这可咋整呢？要是这个`data`就是一系列对象就好了，那就和刚才的实现差不多了，但问题在于，不借助原生数据结构，那么Python中数据是无法自动形成一个系列的。

那我手动把他们变成一个系列就行了。比如两两结对，再两两结对，再……或者在本问题下，问题更简单，一个接着一个就行了，我们实现的是顺序表，它们能记住顺序就可以了。

那我们可以让它们记住自己的编号就可以了，后面索引都方便了。但是如何让一个数据记住自己的编号呢？

用类就行了。

用一个类，它的实例只有两个属性，一个是编号，一个是数据本身。这样数据就有编号了。这样它们就能“记住”自己的顺序了。因此我们首先要实现这个给数据编号的类。

```python
class MyListNode(object):
    def __init__(self, index=0, val=None):
        self._index = index
        self._val = val
```

提醒一下，本来用二元元组就能完成的事情，因为不能用原生数据结构，所以才这样写。实际操作过程中尽量不要写这样的过于简单的类，非常无厘头。

而且，这样的实现方式其实是幼稚的表现。其实想不靠其他数据结构直接实现顺序表非常困难。

比如这样虽然给数据加上了顺序，但是怎么确定它们是属于哪个列表的呢？即使我再加一个属性保存列表名，我依然无法从列表访问这些值。所以，不依靠线性表实现顺序表，是基本上无法做到的。我没有系统地学习逻辑方面的知识，所以无从严格地证明，我只能大概地说明一下我是怎么得出这个结论的

- 要从列表中访问元素值，也就是说列表要保存到元素的访问入口信息
- 如果不确定元素个数，无限制地直接给类加上变量也是不实际的(比如self.a, self.b, ... , self.aa, ...)
- 那么只能保存一个入口，这个入口要么指向一个存储多个对象的容器，要么指向一个对象，再由这个对象可以得到或者推导到其他对象
- 上面前者会用到原生数据结构，后者其实Python里面比较可行的方法是**链表** 

类似的就是，如何在C语言中不使用数组来实现数组。事实上在C中我们可以通过内存操作将开拓的相邻内存地址元素作为列表元素，也就是实现原生的“容器”。在我的知识范围内，并不知道Python中有什么机制能实现这种操作。而通过C拓展就没意思了，这篇文章的主题就是Python自定义类的演示。

一个可行的方式是，我们可以在实现链表以后，再用这个链表实现顺序表。或者我们用类来当作数据容器，通过容器拓展接口实现无限扩展的容器。这两种方式其实是一个意思。那我们就直接写链表吧。于是说好的要写原生列表的，结果变成写链表了。

那我们的节点的结构就要更改了，它要保存自己的数据和下一个元素的链接信息。我们就只做单向链表，不保存前节点的信息。

```python
class LLNode(object):
    """链表的结点"""
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_

    def __repr__(self):
        return '%r' % self.elem
```

好了，现在来处理链表。该链表只需要一个指向第一个元素就可以了，再由元素分别递进指向下一个元素。

```python
class LinkedList(object):
    """单向链表"""
    def __init__(self):
        self._head = None

    @property
    def is_empty(self):
        """判断是否是空链表，返回布尔值"""
        return self._head is None

    def prepend(self, elem):
        """在链表头部插入一个新元素
        TODO:这里面其实还有别的问题我们后面再回来处理
        """
        self._head = LLNode(elem, self._head)

    def __repr__(self):
        cur_node = self._head
        out_str = '['
        while cur_node:
            out_str += '%r' % cur_node.elem
            if cur_node.next:
                out_str += ', '
            cur_node = cur_node.next
        return out_str + ']'
```

其中我们用到了一个装饰器`@property`。这个装饰器的作用是调用这个方法的时候可以像引用实例的属性一样，简单说来就是可以省去`()`，我们可以用`self.is_empty`实现`self.is_empty()`。在前面我已经说过了，这两种方法其实在底层是一样的值访问，只不过形式上做出了区别。

总之，这是我们链表的雏形，目前这两个类就已经可以完成最简单的链表了。

```python
test_list = LinkedList()
node_a = LLNode(1)
test_list.prepend(node_a)  # 插入结点
test_list.prepend('b')  # 插入其他对象
print(test_list)
# 输出 ['b', 1]
```

现在我们再来说`prepend`中说到的“问题”，我们默认是把所有对象都变成`LLNode`的实例插入链表中，但是这会导致一个问题，如果这个对象本来就是一个`LLNode`，那么我们就多此一举了，之所以我们能在最后的输出中正常输出`'b'`和`1`，那是因为我们的`LLNode`类里面的`__repr__`就是返回实例的`elem`属性转换的字符串，但是实际上我们的示例中**`'b'`和`1`是以不同的状态存在链表中的**。一般情况下，我们不愿意把`LLNode`暴露在外面作为用户会接触到的东西，但我们最好还是把`prepend`函数改一下防止以后的操作中出现不必要的麻烦。

```python
def prepend(self, elem):
    """在链表头部插入一个新元素"""
    if isinstance(elem, LLNode):
        elem = elem.elem
    self._head = LLNode(elem, self._head)
```

当然，如果你不想让程序比原来长，那么用一个三元操作符就可以了。

```python
def prepend(self, elem):
    """在链表头部插入一个新元素"""
    self._head = LLNode(elem.elem, self._head) if isinstance(elem, LLNode) else LLNode(elem, self._head)
```

需要说明的是以上两种写法其实都是会造成一定程度的内存的浪费，但是我们是演示而且也不是写C，我就尽量怎么逻辑清楚的同时把代码缩短就行了，不去管内存优化的问题。另一个要说明的地方是为什么我们不直接把已经是LLNode的节点直接插入链表而是要用它的元素重新创建一个节点实例。这是为了避免和列表等可变对象带来的引用对象时类似的暗坑，如果直接引用这些节点，后面我们对节点进行操作的时候就会带来各种不安全的问题。其实也有办法避免直接引用带来的安全问题，这些可能也要放在Python精进部分介绍。

接下来我们要处理链表初始化的问题，比如我们用诸如`list(...)`的方式可以把一些元素直接在创建原生列表的同时插入原生列表。

刚才在插入元素的时候，我们采取的是从列表的头插入新元素，这是因为这种做法比较方便，如果我们要和原生列表一样默认在列表尾插入数据，那么我们需要改造目前的这个链表。按照现在的结构，如果要想在列表尾插入新元素，那么我们首先要找到列表的尾端，也就是根据一个个节点，一个个next过去直到None。如果每次都这样做显然是非常浪费时间的，这也是我刚才选择在列表头插入数据的原因。除了这个办法，还有一种解决方案是在列表中除了头`_head`，我们再加入一个属性记录列表的尾端。

这其实不会浪费太多资源，因为如果我们还是用原来的从头部插入新元素的方法，不会涉及到列表的尾端，不会多出来额外的操作，只有在涉及列表尾端操作的时候才会涉及到。

先不管两种列表内部的方法该如何分配，我们先把这个带尾部信息的链表大概的样子实现出来，用继承可以非常简单地做到。

```python
class LinkedListR(LinkedList):
    """有尾部信息的单向链表"""
    def __init__(self):
        super(LinkedListR, self).__init__()
        self._rear = None

    def prepend(self, elem):
        """在链表头部插入一个新元素"""
        elem = elem.elem if isinstance(elem, LLNode) else elem
        self._head = LLNode(elem, self._head)
        if self.is_empty:
            self._rear = self._head
```

我们对`prepend`方法要进行一些改动，如果列表为空要将列表的尾端初始化。

接下来我们处理在列表尾端插入数据的方法。

```python
def append(self, elem):
    """在链表尾部插入一个新元素"""
    elem = elem.elem if isinstance(elem, LLNode) else elem
    if self.is_empty:
        self._rear = LLNode(elem)
        self._head = self._rear
    else:
        self._rear.next = LLNode(elem)
        self._rear = self._rear.next
```

可以看到加入尾部标记以后在列表尾端插入数据也很简单方便了。而且我们现在如果愿意的话其实需要回来修改一下`prepend`方法。

```python
def prepend(self, elem):
    """在链表头部插入一个新元素"""
    elem = elem.elem if isinstance(elem, LLNode) else elem
    if self.is_empty:
        self._head = LLNode(elem, self._head)
        self._rear = self._head
    else:
        self._head = LLNode(elem, self._head)
```

可以看到程序逻辑没有改变，但是为了和`append`相统一，我们不惜把`self._head = LLNode(elem, self._head)`写了两遍，其实单纯就是为了方便和`append`一起理解，风格一致。当然，这和我以前说的字典最后一个元素后面一样加上`,`是一样的可有可无的小技巧，你不喜欢这样，就喜欢程序简洁一点也是可以的。

我们再来试一下这个加强链表能否正常工作。

```python
test_list = LinkedListR()
node_a = LLNode(1)
node_b = LLNode('b')
test_list.append(node_a)
test_list.append('append')
test_list.prepend('prepend')
test_list.append(node_b)
print(test_list)
# 输出 ['prepend', 1, 'append', 'b']
```

嗯，看上去很正常。现在我们再回来处理刚才说过的链表**初始化**的问题。如果看过本专题前一篇介绍过自定义函数的文章[Python进阶-自定义函数基础](https://www.jianshu.com/p/c535704938eb)我们不难看出来，用`*args`的形式给出初始化参数就可以了，自定义类内部的方法实质上还是自定义函数，只不过用自定义类进行了包装。于是我们`LinkedListR`的`__init__`函数可以改成这样

```python
def __init__(self, *elems):
    super(LinkedListR, self).__init__()
    self._rear = None
    for elem in elems:
        self.append(elem)
```

就是只需要将`*elems`里面的元素依次插入链表即可。我们再来试一下

```python
test_list = LinkedListR(1, '2', 3)
print(test_list)
# 输出[1, '2', 3]
```

至于你说要用`[1, 2, 3]`这样的形式在Python中定义我们的自定义链表，我目前还不知道如何完全用Python代码实现。

至此，我们的链表就真的算是有个雏形了，我们现在回去修改一下前面的代码，总结一下目前的三个类，最后实现一下索引的实现方式作为示例，其他的method我就说几个有代表性的怎么实现，具体代码就不写了。本文已经很长了。

# 链表雏形总结

## 链表节点

```python
class LLNode(object):
    """链表的结点"""
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_

    def __repr__(self):
        return '%r' % self.elem
```

这里说明一下`next_`的拖尾或者说后缀的下划线。我们之前说过前缀下划线代表对象的权限，那么后缀其实就完全是为了防止和内置关键字冲突了。后面我们讲到**迭代器**和**生成器**的时候我们就会接触到`next()`方法，这里就是为了防止和它冲突。

## 单向链表

```python
class LinkedList(object):
    """单向链表"""
    def __init__(self, *elems):
        self._head = None
        for elem in elems:
            self.prepend(elem)

    @property
    def is_empty(self):
        """判断是否是空链表，返回布尔值"""
        return self._head is None

    def prepend(self, elem):
        """在链表头部插入一个新元素"""
        elem = elem.elem if isinstance(elem, LLNode) else elem
        self._head = LLNode(elem, self._head)

    def __repr__(self):
        cur_node = self._head
        out_str = '['
        while cur_node:
            out_str += '%r' % cur_node.elem
            if cur_node.next:
                out_str += ', '
            cur_node = cur_node.next
        return out_str + ']'
```

我们像刚才的带尾标记的链表一样加上了初始化方法，但是默认用的是从头部依次插入数据。

## 带尾标记的链表

```python
class LinkedListR(LinkedList):
    """有尾部信息的单向链表"""
    def __init__(self, *elems):
        super(LinkedListR, self).__init__()
        self._rear = None
        for elem in elems:
            self.append(elem)

    def prepend(self, elem):
        """在链表头部插入一个新元素"""
        elem = elem.elem if isinstance(elem, LLNode) else elem
        if self.is_empty:
            self._head = LLNode(elem, self._head)
            self._rear = self._head
        else:
            self._head = LLNode(elem, self._head)

    def append(self, elem):
        """在链表尾部插入一个新元素"""
        elem = elem.elem if isinstance(elem, LLNode) else elem
        if self.is_empty:
            self._rear = LLNode(elem)
            self._head = self._rear
        else:
            self._rear.next = LLNode(elem)
            self._rear = self._rear.next
```

这里再来讲一下关于继承的问题，我们刚才简单介绍了一下类的继承是什么以及怎么实现。这里借这个例子顺便讲一下两个细节，一个是`__init__`，一个是`prepend`。

我们之所以需要类的继承，是因为继承可以让我们的类具有它的父类的所有属性和方法。如果我们在新的子类中重新定义这个方法或者这个值，那么父类中的对应方法或值就会被覆盖，就像`LinkedListR`里面的`prepend`。我们看到尽管这个方法的逻辑和功能没有太大改变，我们依然需要重写这个方法，为的是**防止新类的新属性值带来的问题**，这是我们设计类的时候经常会忘记的，这也是我建议类内部相似方法的逻辑和代码尽量风格一致的原因，这样我们可以更清楚地明白该类该方法的工作逻辑，减少BUG出现的概率。再说一下`LinkedListR`里面的`__init__`，这也有两个可以说的地方，一个是我们调用父类的`__init__`初始化方法的时候没有把`(*elems)`放进去，这一点是值得注意的，我们要注意调用父类初始化方法的时候要灵活，**要清楚地知道什么样的参数会带来什么样的实例**。另一个就是`super()`，这个函数用来查找一个类的父类中的方法。如果找不到，它会一层层递进基类进行搜索。它一般的调用形式是`super().method(self, ...)`，而我们在这里给super函数两个参数，一个是子类名，一个是实例对象，所以变成了`super(LinkedListR, self)`这样的形式，这样做以后，调用的`__init__`函数就不需要指定`self`了。

总之，我们在设计类的时候，一个是要了解父类的方法属性特点，另一个是要了解子类的方法属性特点。更总结来说，我们要思路清楚、逻辑清晰，区分出子类父类的区别。

# 链表的一些方法实现

首先就是我刚才说过的索引的实现。我们不难猜到，这应该也是由magic method来实现，但是我们看一下原生列表，并没有`__index__`这样的魔术方法(其实这个方法有别的作用)。

```python
>>> dir(list)
['__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
```

我们猜测，索引是由`__getitem__`和`__setitem__`来实现的，从官方文档中能证实我们的猜测。相信从名字也能猜到他们各自负责什么样的功能，一个是读一个是写。所以我们的索引实现起来也很简单。

```python
def __getitem__(self, i):
    """TODO: 负数和切片"""
    if not isinstance(i, int):
        raise TypeError('list indices must be integers')
    cur_node = self._head
    while i:
        try:
            cur_node = cur_node.next
        except AttributeError:
            raise IndexError('list index out of range')
        i -= 1
    if cur_node is None:
        raise IndexError('list index out of range')
    return cur_node.elem
```

这是我们的第一个版本，它可以完成最基本的索引，超出索引范围的时候会报错，我已经试过了没有任何问题。注意到我们返回的时候不应该返回节点而是返回节点里面的数据对象。接下来我们要处理关于负数索引的问题，在当前版本下，负数索引会遍历完链表后抛出`IndexError`。我大概的思路就是，先确定列表的长度，再根据列表长度和负数索引值确认对应的需要遍历的程度。

这里面有个问题，首先我们可能正好可以先去把`len()`方法实现了。但是即使是这样，我们发现链表要想获取长度，必须要进行一次完全遍历。而后我们获取索引值对应的对象，又再要遍历一次。这是由链表的特点所决定的。不像顺序表各元素的位置可以根据内存地址推算出来所以无论是长度还是索引都只需要一次计算，而链表需要重复的操作。这其实是数据结构的知识了，而涉及到插入和删除元素的操作，两者的地位就对换了，这是不同的实现方式带来的操作上的特点。

原则上就像我们刚才让原始链表在头部插入新元素一样，我们完全可以不去实现这样的索引功能，因为如果你需要这样的功能，那么为什么你不用顺序表呢？这是有多种数据结构存在的原因，根据不同的场景就该使用不同的数据结构。所以，我这里就说一下，我们会写`len()`方法，但是负数索引就不写了，因为意义不大，而且写的思路我刚才也已经进行过说明了，实现起来并不困难。而如果要用链表实现快速的负数索引，其实给节点加上一个属性保存前一个节点的信息就行了，这样链表也就变成了**双向链表**，我们并不打算实现这种链表，因为它操作麻烦，每次插入删除数据的操作量是单向链表的两倍，如果只是为了方便实现负数索引，还是不如用顺序表。当然，有兴趣的可以通过继承的方式自己写一下这种链表尝试一下。

`len`方法比较简单，我就直接贴代码不废话了。

```python
def __len__(self):
    cur_node, length = self._head, 0
    while cur_node:
        length += 1
        cur_node = cur_node.next
    return length
```

注意到这个方法是通用的，所以写在`LinkedList`里面。我解释一下为什么要写一个`len`方法而不是直接给类加一个`currentsize`之类的属性，首先是`len`方法虽然经常用到，但并不会被频繁访问，如果加入`size`属性，那么每次列表长度变动的时候都要修改这个属性，代码又难看，效率也不见得高。当然这是我个人的喜好，你完全可以用这种方式实现一下试试。

那我们再来确定切片的处理。首先要说明的是在Python中是有内置的`slice()`类的，所以也会有`slice`对象。这些我在本专题[Python进阶-简单数据结构](https://www.jianshu.com/p/e6c4683a511d)没有说明，因为那时候觉得文章已经太长了，可是现在和本文比起来一点都不长。关于这个类的使用技巧可以查看《Python Cookbook》1.11，也可以参考我的博客[cookbook笔记-切片命名](https://link.jianshu.com?t=http://qotes.top/card/5a0969b9f986a61694bdd29b)。总之，在索引的时候，`[]`内部如果是以`:`分割的数字，那么就会被解释为`slice`切片对象，而具体的起始步进规律和`range`一致，不清楚的可以参考本专题[Python进阶-自定义函数基础](https://www.jianshu.com/p/c535704938eb)文章中，我们实现了手写的`range`函数。下面是考虑到切片以后，`__getitem__`方法的样子。

```python
def __getitem__(self, i):
    cur_node = self._head

    if isinstance(i, slice):
        if not isinstance(i.start, int) and not i.start is None:
            raise TypeError('slice indices must be integers or None')
        if not isinstance(i.stop, int) and not i.stop is None:
            raise TypeError('slice indices must be integers or None')
        if not isinstance(i.step, int) and not i.step is None:
            raise TypeError('slice indices must be integers or None')
        start = i.start or 0
        stop = i.stop or len(self)
        forward = stop - start
        step = i.step or 1
        result = LinkedListR()
        while start:
            try:
                cur_node = cur_node.next
            except AttributeError:
                return result
            start -= 1
        while forward:
            try:
                result.append(cur_node.elem)
                for i_step in range(step):
                    cur_node = cur_node.next
            except AttributeError:
                return result
            forward -= step
        return result

    if not isinstance(i, int):
        raise TypeError('list indices must be integers')
    while i:
        try:
            cur_node = cur_node.next
        except AttributeError:
            raise IndexError('list index out of range')
        i -= 1
    if cur_node is None:
        raise IndexError('list index out of range')
    return cur_node.elem
```

代码一下就长了，而且不是很好读。这个方法也是通用方法，因此可以放在`LinkedList`中。

注意到我们依然屏蔽了负数步进的使用，但是将省略时取到列表头或尾的语法糖实现了。我们试一下

```python
test_list = LinkedListR(1, '2', 3, 4, '5')
print(test_list[::])  # 输出 [1, '2', 3, 4, '5']
print(type(test_list[1]))  # 输出 <class 'str'>
print(type(test_list[:2]))  # 输出 <class '__main__.LinkedListR'>
```

完美。

有了读取，写入操作我们也类似地实现，主要的难点在于对切片的赋值语句该怎么处理，具体的代码有兴趣可以留作练习，我就不写了。

讲完了索引，我再讲一下插入和删除数据的操作。链表插入数据很方便，我们只要找到插入位置，再更改相关节点的链接信息就可以了。在顺序表中我们可能会涉及到讲某元素后面的所有元素平移的操作，会比链表效率低很多。

这里就直接贴代码了。

```python
def insert(self, i, elem):
    """插入数据到指定索引位置"""
    cur_node = self._head
    if not isinstance(i, int):
        raise TypeError('list indices must be integers')
    if not i or not cur_node:
        self.prepend(elem)
        return
    i -= 1
    while i:
        if cur_node.next:
            cur_node = cur_node.next
        else:
            break
        i -= 1
    new_node = LLNode(elem, cur_node.next)
    cur_node.next = new_node
```

基本上和内置列表的逻辑一致，超出索引范围后直接加在链表的尾部，测试结果

```python
test_list = LinkedListR(1, '2', 3, 4, '5')
test_list.insert(0, 'head_insert')
print(test_list)  # 输出 ['head_insert', 1, '2', 3, 4, '5']
test_list.insert(2, 'index_insert')
print(test_list)  # 输出 ['head_insert', 1, 'index_insert', '2', 3, 4, '5']
test_list.insert(20, 'rear_insert')
print(test_list)  # 输出 ['head_insert', 1, 'index_insert', '2', 3, 4, '5', 'rear_insert']
```

但是这个是通用的`insert`方法，我们没有对`_rear`进行处理。我们简单判断新节点是不是尾节点即可。

```python
def insert(self, i, elem):
    """插入数据到指定索引位置"""
    cur_node = self._head
    if not isinstance(i, int):
        raise TypeError('list indices must be integers')
    if not i or not cur_node:
        self.prepend(elem)
        return
    i -= 1
    while i:
        if cur_node.next:
            cur_node = cur_node.next
        else:
            break
        i -= 1
    new_node = LLNode(elem, cur_node.next)
    cur_node.next = new_node
    if not new_node.next:
        self._rear = new_node
```

接下来我就写最后一个具体的例子就是关于加法。这可以通过内置的魔术方法`__add__`来实现。我们以后可以直接对`LinkedListR`进行加法运算，比如说`LinkedListR(1, 2) + LinkedListR(3, 4)`就能直接得到`LinkedListR(1, 2, 3, 4)`。

```python
def __add__(self, another):
    if not isinstance(another, LinkedListR):
        raise TypeError('can only concatenate list (not "%r") to list' % type(another))
    if self.is_empty:
        return another
    self._rear.next = another._head
    self._rear = another._rear
    return self
```

由于之前说明过两种链表逻辑上的区别，所以我并不打算给`LinkedList`写加法。

我们稍加修改，还能实现类似`collections`里面`deque`的`extend`方法，这里就不写了。

另外的一些方法，比如说`__iter__`，我们还没讲过迭代器，所以不打算讲。还有比较运算符和交并异或运算符，都是很简单的判断类型判断长度依次操作的方法，也不展开讲了。还有一些hook和subclass回溯相关的内容我也不讲了，有机会在Python精进中找个地方介绍一下。而像`sort`、`pop`之类的功能性方法实现起来难度也不大，有兴趣可以自己实现一下试试看。

# 总结

最后来个全家福。

```python
class LLNode(object):
    """链表的结点"""
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_

    def __repr__(self):
        return '%r' % self.elem


class LinkedList(object):
    """单向链表"""
    def __init__(self, *elems):
        self._head = None
        if len(elems) == 1:
            if isinstance(elems[0], Iterable):
                for elem in elems[0]:
                    self.prepend(elem)
            else:
                self.prepend(elems[0])
        else:
            for elem in elems:
                self.prepend(elem)

    @property
    def is_empty(self):
        """判断是否是空链表，返回布尔值"""
        return self._head is None

    def prepend(self, elem):
        """在链表头部插入一个新元素"""
        elem = elem.elem if isinstance(elem, LLNode) else elem
        self._head = LLNode(elem, self._head)

    def __repr__(self):
        cur_node = self._head
        out_str = '['
        while cur_node:
            out_str += '%r' % cur_node.elem
            if cur_node.next:
                out_str += ', '
            cur_node = cur_node.next
        return out_str + ']'

    def __len__(self):
        cur_node, length = self._head, 0
        while cur_node:
            length += 1
            cur_node = cur_node.next
        return length

    def __getitem__(self, i):
        cur_node = self._head

        if isinstance(i, slice):
            if not isinstance(i.start, int) and not i.start is None:
                raise TypeError('slice indices must be integers or None')
            if not isinstance(i.stop, int) and not i.stop is None:
                raise TypeError('slice indices must be integers or None')
            if not isinstance(i.step, int) and not i.step is None:
                raise TypeError('slice indices must be integers or None')
            start = i.start or 0
            stop = i.stop or len(self)
            forward = stop - start
            step = i.step or 1
            result = LinkedListR()
            while start:
                try:
                    cur_node = cur_node.next
                except AttributeError:
                    return result
                start -= 1
            while forward:
                try:
                    result.append(cur_node.elem)
                    for i_step in range(step):
                        cur_node = cur_node.next
                except AttributeError:
                    return result
                forward -= step
            return result

        if not isinstance(i, int):
            raise TypeError('list indices must be integers')
        while i:
            try:
                cur_node = cur_node.next
            except AttributeError:
                raise IndexError('list index out of range')
            i -= 1
        if cur_node is None:
            raise IndexError('list index out of range')
        return cur_node.elem

    def insert(self, i, elem):
        """插入数据到指定索引位置"""
        cur_node = self._head
        if not isinstance(i, int):
            raise TypeError('list indices must be integers')
        if not i:
            self.prepend(elem)
            return
        i -= 1
        while i:
            if cur_node.next:
                cur_node = cur_node.next
            else:
                break
            i -= 1
        new_node = LLNode(elem, cur_node.next)
        cur_node.next = new_node


class LinkedListR(LinkedList):
    """有尾部信息的单向链表"""
    def __init__(self, *elems):
        super(LinkedListR, self).__init__()
        self._rear = None
        if len(elems) == 1:
            if isinstance(elems[0], Iterable):
                for elem in elems[0]:
                    self.append(elem)
            else:
                self.append(elems[0])
        else:
            for elem in elems:
                self.append(elem)

    def prepend(self, elem):
        """在链表头部插入一个新元素"""
        elem = elem.elem if isinstance(elem, LLNode) else elem
        if self.is_empty:
            self._head = LLNode(elem, self._head)
            self._rear = self._head
        else:
            self._head = LLNode(elem, self._head)

    def append(self, elem):
        """在链表尾部插入一个新元素"""
        elem = elem.elem if isinstance(elem, LLNode) else elem
        if self.is_empty:
            self._rear = LLNode(elem)
            self._head = self._rear
        else:
            self._rear.next = LLNode(elem)
            self._rear = self._rear.next

    def insert(self, i, elem):
        """插入数据到指定索引位置"""
        cur_node = self._head
        if not isinstance(i, int):
            raise TypeError('list indices must be integers')
        if not i or not cur_node:
            self.prepend(elem)
            return
        i -= 1
        while i:
            if cur_node.next:
                cur_node = cur_node.next
            else:
                break
            i -= 1
        new_node = LLNode(elem, cur_node.next)
        cur_node.next = new_node
        if not new_node.next:
            self._rear = new_node

    def __add__(self, another):
        if not isinstance(another, LinkedListR):
            raise TypeError('can only concatenate list (not "%r") to list' % type(another))
        if self.is_empty:
            return another
        self._rear.next = another._head
        self._rear = another._rear
        return self
```

注意到我把`__init__`函数改了一下，因为我想起来`list()`是接受的可迭代对象作为参数。我在不影响目前的逻辑下让我们的链表既可以直接接受多个元素进来，也可以接受一个可迭代对象把元素都放进来。

我们总算半完成了这个“浩大”的工程，但是里面还是有一些问题值得改进的：

1. 我们的链表头尾初始化值都设置成了`None`，设置成`LLNode(None)`会不会更好呢？
2. 我们设计一些方法的时候该不该考虑兼容`LinkedList`和`LinkedListR`进行互相操作的场景和实现这种可能性呢？
3. 我们设计节点的时候定义了一个简单的类，是不是可以用字典来实现呢？这两种方案会有什么区别？
4. 我们能不能针对原生列表或者元组借此进行一些性能的优化呢？
5. 。。。

回答

1. 会，~~但是我后来发现的时候懒得改了~~。
2. 该，但是代码会比较复杂不适合演示，~~其实是懒得写~~。
3. 可以，~~但是不清真~~。逻辑不那么清晰，操作、重构起来更麻烦。
4. 可以，~~但是为什么不写C拓展呢~~。
5. 。。。

以上这些问题或多或少是和编程范式相关的内容，简单的一个线性表，我们用了这么长的篇幅。这是我不愿意讲数据结构的原因。要知道我们还只是介绍线性表的实现，而线性表的使用才是该数据结构的重点，我们只是在实现的过程中粗略介绍了链表和顺序表各种操作的消耗多少，真的要讲线性表按我这个啰嗦的样子大概还能讲十几万字。而且我们没有介绍过算法，因此我没有讲**复杂度**，很多地方我只能说一个大概的比较，谁操作起来更快，而不是说顺序表索引的复杂度是O(1)，链表的索引复杂度是O(n)，顺序表插入的复杂度是O(n)，链表插入的复杂度是O(1)这样的话让懂的人觉得我在说废话让不懂的人一头雾水。

除此之外还有一些问题比如说Python字符串用连接的方式处理不是很方便，有没有别的办法可以优化`__repr__`的实现。我们写的这个链表还有很多地方值得修改。

总之，本文的重点还是在介绍Python的自定义类的一些基本用法，也劝退了自己开数据结构板块的想法。

当然，我们说过链表比顺序表在插入数据的时候更有优势。那么我们写的这个链表在插入数据的时候比起用C写的原生列表如何呢？我们来试一下

```python
from time import time
test_list = LinkedListR()
t = time()
for i in range(100000):
    test_list.insert(100, i)
print(time() - t)  # 输出 1.6076409816741943

test_list = []
t = time()
for i in range(100000):
    test_list.insert(100, i)
print(time() - t)  # 输出 2.5222737789154053
```

经过我自己的电脑的输出测试，在列表较小的时候，比如说10000次操作，我们的链表大概需要0.15s而原生顺序表只需要0.02s。但是来到100000次操作，我们的链表操作时间几乎就是乘以10变成了1.6s，而顺序表的时间增长了不知道多少倍(大概100倍)来到了2.5s，我们的链表已经比原生列表更“快”了。

要知道，这还是在前提是我们的链表是用Python实现，而CPython的列表是C实现的。Python几乎是以“慢”著称的，而C则是性能标杆，我们能实现“超越”就已经是数据结构和算法上的优势太明显了。

这也是我们中途放弃写顺序表改写链表的原因，纯粹地实现一个顺序表不仅难度非常大，而且没什么意义。我们在实现这个链表的过程中，没有用到任何的列表或者元组(显示地)和字典，所有都靠自定义类来抽象完成。

这难道不有趣吗？

# 补充

由于篇幅过长，本来应该讲的除了`@property`以外两个更重要的修饰器`@staticmethod`和`@classmethod`反而没讲，原因在于我们的演示里面实在是用不到这两个装饰器。但是平时使用过程中，这两个修饰器出场率还是很高的，对初学者来说理解起来也会有一定的混淆。我就简单地补充介绍一下吧。

`staticmethod`是用来修饰一个包装在类内部的不需要实例来调用的静态方法的装饰器。比如说我们刚才的`Cat`，我们定义几个函数对比一下

```python
class Cat(object):
    def __init__(self):
        pass

    def meow(self):
        print('mewo')

    @staticmethod
    def meowmeow():
        print('meowmeow')

    @classmethod
    def meowmeowmeow(cls):
        cls.meowmeow()
```

注意到`meowmeow`和`meow`唯一的区别在于前者不需要`self`这个参数。这意味着我们在`import`这个类以后，可以直接调用`Cat.meowmeow()`。但是`meow`不可以，它需要一个实例以`catinstance.meow()`的形式调用。

接下来是`classmethod`，这个装饰器的特点在于第一个参数是类。通过`meowmeowmeow`和`meow`两个函数对比可以看出来区别，如果是`self`我们应该调用的是`self.meow()`，但是刚才说过，`staticmethod`让我们可以直接用类调用这样的方法。值得一提的是`classmethod`也可以用`Cat.meowmeowmeow()`这样的形式调用。

而这三种方法，都可以被实例调用，也就是说如果`c = Cat()`，那么`c.meow()`、`c.meowmeow()`和`c.meowmeowmeow()`都是可以的。

# 后话

写完才发现可能本文长得有些过分了。之前都尽量控制在一万字以内，这次代码比较长，所以可能会大概有一万五。

如果你能坚持看完，那么我是很佩服你的。

中间写链表写high了可能有一些细节忘记讲了，要是以后能想起来再补充吧。

下一篇我可能就会开始介绍一下关于编程范式的问题了，本文讲了自定义类，但是并没有着重介绍面向对象的思想观念，主要是为了打好基础，熟悉一下自定义类设计方法以及实现的时候会用到的一些套路和技巧。这也是本专题叫作“爬着学Python”的原因。单纯地`import requests`下几张妹子图不叫学爬虫。我们以后接触到scrapy的时候会意识到用面向对象的方式写爬虫框架是多么的方便，可以写出健壮的，可复用可迭代的爬虫。