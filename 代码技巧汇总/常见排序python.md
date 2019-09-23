# 排序算法相关概念

- **稳定**：如果a原本在b前面，而a=b，排序之后a仍然在b的前面； 
     
- **不稳定**：如果a原本在b的前面，而a=b，排序之后a可能会出现在b的后面；
     
- **内排序**：所有排序操作都在内存中完成； 
     
- **外排序**：由于数据太大，因此把数据放在磁盘中，而排序通过磁盘和内存的数据传输才能进行；
     

## 常见排序

### 冒泡排序

冒泡排序重复地走访过要排序的数列，一次比较两个元素，如果他们的顺序错误就把他们交换过来。走访数列的工作是重复地进行直到没有再需要交换。因为每次遍历，最大的元素都会被送到最右端，故名冒泡排序。

**步骤**：

1. 比较相邻的元素。如果第一个比第二个大，就交换他们两个。
2. 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。在这一点，最后的元素应该会是最大的数。
3. 针对所有的元素重复以上的步骤，除了最后一个。
4. 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。

**代码实现**：

```python
def bubble_sort(nums):
    size = len(nums)
    for i in range(size):
        for j in range(size-i-1):
            if nums[j] > nums[j+1]:
                nums[j+1], nums[j] = nums[j], nums[j+1]
    return nums
```

我们可以考虑设置一标志性变量pos，用于记录每趟排序中最后一次进行交换的位置。由于pos位置之后的记录均已交换到位，故在进行下一趟排序时只要扫描到pos位置即可。

改进后的冒泡排序：

```python
def bubble_sort2(nums):
    size = len(nums)
    i = size - 1
    while i > 0:
        pos = 0
        for j in range(i):
            if nums[j] > nums[j+1]:
                pos = j
                nums[j], nums[j+1] = nums[j+1], nums[j]
        i = pos
    return nums
```

**冒泡排序动图演示：**



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-19-042537.jpg)



### 选择排序

选择排序(Selection-sort)是一种简单直观的排序算法。它的工作原理是：首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

代码实现如下：

```python
def select_sort(nums):
    size = len(nums)
    for i in range(size-1):
        # 找出最小的数
        min_index = i
        for j in range(i+1, size):
            if nums[j] < nums[min_index]:
                min_index = j
        nums[i], nums[min_index] = nums[min_index], nums[i]
    return nums
```

选择排序动图演示如下：



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-19-042538.jpg)



### 插入排序

插入排序（Insertion-Sort）的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。

具体步骤：

- 从第一个元素开始，该元素可以认为已经被排序；
- 取出下一个元素，在已经排序的元素序列中从后向前扫描；
- 如果该元素（已排序）大于新元素，将该元素移到下一位置；
- 重复步骤3，直到找到已排序的元素小于或者等于新元素的位置；
- 将新元素插入到该位置后；
- 重复步骤2~5。

**代码实现：**

```python
def insertion_sort(nums):
    size = len(nums)
    for i in range(1, size):
        cur_val = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > cur_val:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = cur_val  # 找到位置进行插入
    return nums
```

可以考虑使用二分查找来寻找插入的位置：

```python
def insertion_sort2(nums):
    size = len(nums)
    for i in range(1, size):
        val = nums[i]
        left, right = 0, i-1
        while left <= right:
            mid = (left+right)//2
            if val < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        for j in range(i-1, left-1, -1):
            nums[j+1] = nums[j]
        nums[left] = val
    return nums
```

插入排序动图演示如下：



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-19-042535.jpg)



### 归并排序

归并排序是建立在归并操作上的一种有效的排序算法。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用。归并排序是一种稳定的排序方法。将已有序的子序列合并，得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。若将两个有序表合并成一个有序表，称为2-路归并。

代码实现如下：

```python
def merge_sort(nums):
    size = len(nums)
    if size < 2:
        return nums
    mid = size//2
    left, right = nums[:mid], nums[mid:]
    return merge(merge_sort(left), merge_sort(right))

def merge(left, right):
    res = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1

    while i < len(left):
        res.append(left[i])
        i += 1
    while j < len(right):
        res.append(right[j])
        j += 1
    return res
```

其动图演示如下：



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-19-042539.jpg)



### 快速排序

快速排序是处理大数据最快的排序算法之一。它的基本思想是，通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。

快速排序基本步骤：

- 从数列中挑出一个元素，称为 "基准"（pivot）；
- 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
- 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。

其代码实现如下：

```python
def partition(nums, left, right):
    pivot = left   # 使用左端元素作为基准
    for i in range(left+1, right+1):
        if nums[i] < nums[left]:
            pivot += 1
            nums[i], nums[pivot] = nums[pivot], nums[i]
    nums[left], nums[pivot] = nums[pivot], nums[left]
    return pivot


def quick_sort(nums, left=0, right=None):
    if right is None:
        right = len(nums) - 1

    def quick_sort_helper(nums, left, right):
        if left >= right:
            return
        pivot = partition(nums, left, right)
        quick_sort_helper(nums, left, pivot-1)
        quick_sort_helper(nums, pivot+1, right)

    return quick_sort_helper(nums, left, right)
```

如果不要求在原地修改数组：

```python
def quick_sort2(arr):
    if len(arr) <= 1:
        return arr
    else:
        return quick_sort2([x for x in arr[1:] if x < arr[0]]) + \
            [arr[0]] + \
            quick_sort2([x for x in arr[1:] if x >= arr[0]])
```

快速排序的动图演示如下：



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-19-042536.jpg)



后面的几种排序方法比较少见，仅在概念上进行讲解。

### 希尔排序

先将整个待排元素序列分割成若干子序列（由相隔某个“增量”的元素组成的）分别进行直接插入排序，然后依次缩减增量再进行排序，待整个序列中的元素基本有序（增量足够小）时，再对全体元素进行一次直接插入排序（增量为1）。

### 堆排序

堆排序（Heapsort）是指利用堆这种数据结构所设计的一种排序算法。堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：即子结点的键值或索引总是小于（或者大于）它的父节点。

步骤：

1.  创建最大堆:将堆所有数据重新排序，使其成为最大堆
     
2.  最大堆调整:作用是保持最大堆的性质，是创建最大堆的核心子程序
     
3.  堆排序:移除位在第一个数据的根节点，并做最大堆调整的递归运算
     

### 计数排序

计数排序使用一个额外的数组`C`，其中第`i`个元素是待排序数组A中值等于`i`的元素的个数。然后根据数组`C`来将`A`中的元素排到正确的位置。

算法的步骤如下：

1. 找出待排序的数组中最大和最小的元素
2. 统计数组中每个值为`i`的元素出现的次数，存入数组`C`的第`i`项
3. 对所有的计数累加（从`C`中的位置为`1`的元素开始，每一项和前一项相加）
4. 反向填充目标数组：将每个元素`i`放在新数组的第`C(i)`项，每放一个元素就将`C(i)`减去1

由于用来计数的数组`C`的长度取决于待排序数组中数据的范围（等于待排序数组的最大值与最小值的差加上1），这使得计数排序对于数据范围很大的数组，需要大量时间和内存。

### 桶排序

桶排序 (Bucket sort)或所谓的箱排序，是一个排序算法，工作的原理是将数组分到有限数量的桶子里。每个桶子再个别排序（有可能再使用别的排序算法或是以递归方式继续使用桶排序进行排序）

桶排序以下列程序进行：

1.  设置一个定量的数组当作空桶子。
     
2.  寻访串行，并且把项目一个一个放到对应的桶子去。（hash）
     
3.  对每个不是空的桶子进行排序。
     
4.  从不是空的桶子里把项目再放回原来的串行中。
     

### 基数排序

基数排序（英语：Radix sort）是一种非比较型整数排序算法，其原理是将整数按位数切割成不同的数字，然后按每个位数分别比较。由于整数也可以表达字符串（比如名字或日期）和特定格式的浮点数，所以基数排序也不是只能使用于整数。

它是这样实现的：将所有待比较数值（正整数）统一为同样的数位长度，数位较短的数前面补零。然后，从最低位开始，依次进行一次排序。这样从最低位排序一直到最高位排序完成以后, 数列就变成一个有序序列。

### 延伸-外部排序

从是否使用外存方面来看，我们可以将排序算法分为内部排序和外部排序：



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-19-042540.jpg)



上面讲的十种排序算法都属于内部排序算法，也就是排序的整个过程都在内存中完成。而当待排序的文件比内存的可使用容量还大时，文件无法一次性放到内存中进行排序，需要借助于外部存储器（例如硬盘、U盘、光盘），这时就需要用到外部排序算法来解决。下面简单介绍一下外部排序算法。

外部排序算法由两个阶段构成：

1. 按照内存大小，将大文件分成若干长度为 l 的子文件（l 应小于内存的可使用容量），然后将各个子文件依次读入内存，使用适当的内部排序算法对其进行排序（排好序的子文件统称为“归并段”或者“顺段”），将排好序的归并段重新写入外存，为下一个子文件排序腾出内存空间；
2. 对得到的顺段进行合并，直至得到整个有序的文件为止。

例如，我们要对一个大文件（无法放进内存）进行排序，可以将其分成多个大小可以放进内存的临时文件，然后将这些较小的临时文件依次进入内存，采取适当的内存排序算法对其中的记录进行排序，将得到的有序文件（初始归并段）移至外存；之后再对这些排序好的临时文件两两归并，直至得到一个完整的有序文件。如下图所示：



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-19-42537.jpg)



### 总结



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-19-42540.jpg)



其中，`n`表示数据规模，`k`表示桶的个数，`In-place`表示不占用额外内存，`Out-place`表示占用额外内存。