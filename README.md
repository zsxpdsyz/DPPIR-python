# DPPIR-python
# Makefile

[使用变量 - 跟我一起写Makefile 1.0 文档](https://seisman.github.io/how-to-write-makefile/variables.html)

### unifuzzer的make工程代码

```makefile
# CC指定编译器
CC := clang
# cflags表示编译选项，-wall表示查看警告信息
CFLAGS := -Wall -fPIC
#CFLAGS += -O2
#表示在原来的功能上加上一些东西
CFLAGS += -g -DUF_DEBUG
#优化参数，比如说指定需要链接的库文件的位置
LDFLAGS := -fsanitize=fuzzer -lunicorn -pthread

OUT := uf
# 指定默认要编译的对象，如果没有指定，make只会默认编译他遇到的第一个对象
.DEFAULT_GOAL := all
# wildcard表示将后面的通配符全部展开。因为在变量定义和函数引用的时候通配符会失效。
SRC := $(wildcard callback/*.c) \
		$(wildcard uniFuzzer/*.c) \
		$(wildcard uniFuzzer/elfLoader/*.c)
OBJ := $(SRC:.c=.o)

MAIN_SRC := uniFuzzer/uniFuzzer.c
MAIN_OBJ := uniFuzzer/uniFuzzer.o
# 表示flag只会作用到MAIN_OBJ编译引发的规则中
$(MAIN_OBJ): CFLAGS += -fsanitize=fuzzer -IuniFuzzer/elfLoader
# 去掉SRC中MAIN_SRC的部分，并保留剩下的结果
OTHER_SRC := $(filter-out $(MAIN_SRC),$(SRC))
# 把OTHER_SRC变量中所有以.c结尾的变量换成以.o结尾
OTHER_OBJ := $(OTHER_SRC:.c=.o)
# $^表示之前提到的所有依赖的值，如$(OBJ)
# $@: 对应的目标文件
# $<: 第一个依赖文件的名称
# 下面表示提取出的公式，即：所有.o文件的生成依赖于所有.c文件。%为通配符，在同一句话中，%表示的是同一个名字。
# 这里的通配符%，指的是匹配所有满足条件的target，而不是匹配该文件夹下所有的满足条件的文件
%.o:%.c
	$(CC) -o $@ $(CFLAGS) -c $<

all:$(OUT)

$(OUT):$(OBJ)
	$(CC) -o $@ $(LDFLAGS) $^

clean:
	rm -f $(OUT) $(OBJ)
# 伪目标。如果不把all加入进去的话，make只会执行第一个目标和依赖。clean后面没跟依赖项
# 说明是单纯按照脚本来执行的。
.PHONY: all clean
```
