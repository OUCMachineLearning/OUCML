# gcc g 常用命令

[Edit](https://github.com/hokein/Wiki/wiki/gcc-g--常用命令/_edit)[New Page](https://github.com/hokein/Wiki/wiki/_new)

hokein edited this page on 5 Feb 2013 · [1 revision](https://github.com/hokein/Wiki/wiki/gcc-g--常用命令/_history)

```
//生成.o链接文件
gcc -c main.c

//生成编译过程中间所有文件(.i, .s, .o)
gcc -save-temps main.c

//由.o文件生成静态链接库
ar crv libmain.a main.o

//编译链接静态链接库,生成可执行文件
gcc test.c libmain.a -o test 

//生成动态链接库
gcc main.cpp -fPIC -shared -o libmain.so
gcc -fPIC -shared main.o -o libmain.so

//链接动态链接库，生成可执行文件:
gcc test.c -L. -lmain -o test
```

由于编译器只会在特定目录下查找.so动态链接库， 如果要链接自定义的库，可以采取下面做法：

1. copy到usr/lib目录下
2. 设置`LD_LIBRARY_PATH`环境变量：

```
    export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"path/to/lib"
```

1. 通过-Wl,-rpath, path/to/lib参数指定：

```
    gcc test.c -L. -lmain -Wl,-rpath, path/to/lib
```