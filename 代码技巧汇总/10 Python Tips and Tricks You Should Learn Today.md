# 10 Python Tips and Tricks You Should Learn Today

转载自:https://towardsdatascience.com/10-python-tips-and-tricks-you-should-learn-today-a05c23a39dc5

![image-20191201125814645](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-01-045815.png)

(注:高中级别英语,不翻译了哈)

According to Stack Overflow, Python is the fastest growing programming language. The latest [report from Forbes](https://www.whatech.com/development/press-release/442278-why-developers-vote-python-as-the-best-application-programming-language) states that Python showed a 456-percent growth in last year. Netflix uses Python, IBM uses Python, and hundreds of other companies all use Python. Let’s not forget Dropbox. Dropbox is also created in Python. According to [research by Dice](https://insights.dice.com/2016/02/01/whats-hot-and-not-in-tech-skills/) Python is also one of the hottest skills to have and also the most popular programming language in the world based on the [Popularity of Programming Language Index](https://pypl.github.io/PYPL.html).

Some of the advantages Python offers when compared to other programming languages are:

1. Compatible with major platforms and operating systems
2. Many open-source frameworks and tools
3. Readable and maintainable code
4. Robust standard library
5. Standard test-driven development

------

# Python Tips and Tricks

In this piece, I’ll present 10 useful code tips and tricks that can help you in your day-to-day tasks. So without further ado, let’s get started.

# 1. Concatenating Strings(连接字符串)

When you need to concatenate a list of strings, you can do this using a *for loop* by adding each element one by one. However, this would be very inefficient, especially if the list is long. In Python, strings are immutable, and thus the left and right strings would have to be copied into the new string for every pair of concatenation.

A better approach is to use the `join()` function as shown below:

```
characters = ['p', 'y', 't', 'h', 'o', 'n']
word = "".join(characters)
print(word) # python
```

------

# 2. Using List Comprehensions(列表解析语法)

List comprehensions are used for creating new lists from other iterables. As list comprehensions returns lists, they consist of brackets containing the expression, which is executed for each element along with the for loop to iterate over each element. List comprehension is faster because it is optimized for the Python interpreter to spot a predictable pattern during looping.

As an example let’s find the squares of the first five whole numbers using list comprehensions.

```
m = [x ** 2 for x in range(5)]
print(m) # [0, 1, 4, 9, 16]
```

Now let’s find the common numbers from two list using list comprehension

```
list_a = [1, 2, 3, 4]
list_b = [2, 3, 4, 5]
common_num = [a for a in list_a for b in list_b if a == b]
print(common_num) # [2, 3, 4]
```

------

# 3. Iterate With `enumerate()`

Enumerate() method adds a counter to an iterable and returns it in a form of enumerate object.

Let’s solve the classic coding interview question named popularly as the Fizz Buzz problem.

> Write a program that prints the numbers in a list, for multiples of ‘3’ print “fizz” instead of the number, for the multiples of ‘5’ print “buzz” and for multiples of both 3 and 5 it prints “fizzbuzz”.

```
numbers = [30, 42, 28, 50, 15]
for i, num in enumerate(numbers):
    if num % 3 == 0 and num % 5 == 0:
       numbers[i] = 'fizzbuzz'
    elif num % 3 == 0:
       numbers[i] = 'fizz'
    elif num % 5 == 0:
       numbers[i] = 'buzz'
print(numbers) # ['fizzbuzz', 'fizz', 28, 'buzz', 'fizzbuzz']
```

------

# 4. Using ZIP When Working with Lists(使用ZIP方法关联List)

Suppose you were given a task to combine several lists with the same length and print out the result? Again, here is a more generic way to get the desired result by utilizing `zip()`as shown in the code below:

```
countries = ['France', 'Germany', 'Canada']
capitals = ['Paris', 'Berlin', 'Ottawa']
for country, capital in zip(countries,capitals):
    print(country, capital) # France Paris 
                              Germany Berlin
                              Canada Ottawa
```

------

# 5. Using itertools(使用迭代器)

The Python `itertools` module is a collection of tools for handling iterators. `itertools` has multiple tools for generating iterable sequences of input data. Here I will be using `itertools.combinations()` as an example. `itertools.combinations()` is used for building combinations. These are also the possible groupings of the input values.

Let’s take a real world example to make the above point clear.

> Suppose there are four teams playing in a tournament. In the league stages every team plays against every other team. Your task is to generate all the possible teams that would compete against each other.

Let’s take a look at the code below:

```
import itertools
friends = ['Team 1', 'Team 2', 'Team 3', 'Team 4']
list(itertools.combinations(friends, r=2)) # [('Team 1', 'Team 2'),      ('Team 1', 'Team 3'),  ('Team 1', 'Team 4'),  ('Team 2', 'Team 3'),  ('Team 2', 'Team 4'),  ('Team 3', 'Team 4')]
```

The important thing to notice is that order of the values doesn’t matter. Because `('Team 1', 'Team 2')` and `('Team 2', 'Team 1')` represent the same pair, only one of them would be included in the output list. Similarly we can use `itertools.permutations()` as well as other functions from the module. For a more complete reference, check out [this amazing tutorial](https://medium.com/@jasonrigden/a-guide-to-python-itertools-82e5a306cdf8).

------

# 6. Using Python Collections(使用python容器)

Python collections are container data types, namely lists, sets, tuples, dictionary. The collections module provides high-performance datatypes that can enhance your code, making things much cleaner and easier. There are a lot of functions provided by the collections module. For this demonstration, I will be using `Counter()` function.

The `Counter()` function takes an iterable, such as a list or tuple, and returns a Counter Dictionary. The dictionary’s keys will be the unique elements present in the iterable, and the values for each key will be the count of the elements present in the iterable.

To create a `counter` object, pass an iterable (list) to `Counter()` function as shown in the code below.

```
from collections import Countercount = Counter(['a','b','c','d','b','c','d','b'])
print(count) # Counter({'b': 3, 'c': 2, 'd': 2, 'a': 1})
```

For a more complete reference, check out my [python collections tutorial](https://towardsdatascience.com/a-hands-on-guide-to-python-collections-aa350cb399e3).

------

# 7. Convert Two Lists Into a Dictionary(利用ZIP和Dict构建字典)

Let’s say we have two lists, one list contains names of the students and second contains marks scored by them. Let’s see how we can convert those two lists into a single dictionary. Using the zip function, this can be done using the code below:

```
students = ["Peter", "Julia", "Alex"]
marks = [84, 65, 77]
dictionary = dict(zip(students, marks))
print(dictionary) # {'Peter': 84, 'Julia': 65, 'Alex': 77}
```

------

# 8. Using Python Generators(使用python生成器代替 list)

Generator functions allow you to declare a function that behaves like an iterator. They allow programmers to make an iterator in a fast, easy, and clean way. Let’s take an example to explain this concept.

> Suppose you’ve been given to find the sum of the first 100000000 perfect squares, starting with 1.

Looks easy right? This can easily be done using list comprehension but the problem is the large inputs size. As an example let’s take a look at the below code:

```
t1 = time.clock()
sum([i * i for i in range(1, 100000000)])
t2 = time.clock()
time_diff = t2 - t1
print(f"It took {time_diff} Secs to execute this method") # It took 13.197494000000006 Secs to execute this method
```

On increasing the perfect numbers we need to sum over, we realize that this method is not feasible due to higher computation time. Here’s where Python Generators come to the rescue. On replacing the brackets with parentheses we change the list comprehension into a generator expression. Now let’s calculate the time taken:

```
t1 = time.clock()
sum((i * i for i in range(1, 100000000)))
t2 = time.clock()
time_diff = t2 - t1
print(f"It took {time_diff} Secs to execute this method") # It took 9.53867000000001 Secs to execute this method
```

As we can see, time taken has been reduced quite a bit. This effect will be even more pronounced for larger inputs.

For a more complete reference, check out my article [Reduce Memory Usage and Make Your Python Code Faster Using Generators](https://towardsdatascience.com/reduce-memory-usage-and-make-your-python-code-faster-using-generators-bd79dbfeb4c).

------

# 9. Return Multiple Values From a Function(函数多返回值)

Python has the ability to return multiple values from a function call, something missing from many other popular programming languages. In this case the return values should be a comma-separated list of values and Python then constructs a *tuple* and returns this to the caller. As an example see the code below:

```
def multiplication_division(num1, num2):
    return num1*num2, num1/num2product, division = multiplication_division(15, 3)
print("Product=", product, "Quotient =", division) # Product= 45 Quotient = 5.0
```

------

# 10. Using sorted() Function(使用排序方法)

Sorting any sequence is very easy in Python using the built-in method `sorted()`which does all the hard work for you. `sorted()`sorts any sequence (list, tuple) and always returns a list with the elements in sorted manner. Let’s take an example to sort a list of numbers in ascending order.

```
sorted([3,5,2,1,4]) # [1, 2, 3, 4, 5]
```

Taking another example, let’s sort a list of strings in descending order.

```
sorted(['france', 'germany', 'canada', 'india', 'china'], reverse=True) # ['india', 'germany', 'france', 'china', 'canada']
```

------

# Conclusions(总结)

In this article, I’ve presented 10 Python tips and tricks that can be used as a reference in your day-to-day work. I hope you enjoyed this article. Stay tuned for my next piece, “Tips and Tricks to Speed Up Your Python Code”.