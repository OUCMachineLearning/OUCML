# Bash Shell常用快捷键

## 移动光标

- `ctrl+b`: 前移一个字符(backward)
- `ctrl+f`: 后移一个字符(forward)
- `alt+b`: 前移一个单词
- `alt+f`: 后移一个单词
- `ctrl+a`: 移到行首（a是首字母）
- `ctrl+e`: 移到行尾（end）
- `ctrl+xx`: 行首到当前光标替换

## 编辑命令

- `alt+.`: 粘帖最后一次命令最后的参数（通常用于`mkdir long-long-dir`后, `cd`配合着`alt+.`）
- `alt+d`: 删除当前光标到临近右边单词开始(delete)
- `ctrl+u`: 删除光标左边所有
- `ctrl+h`: 删除光标前一个字符（相当于backspace）
- `ctrl+d`: 删除光标后一个字符（相当于delete）
- `ctrl+w`: 删除当前光标到临近左边单词结束(word)
- `ctrl+k`: 删除光标右边所有
- `ctrl+l`: 清屏
- `ctrl+shift+c`: 复制（相当于鼠标左键拖拽）
- `ctrl+shift+v`: 粘贴（相当于鼠标中键）

## 其它

- `ctrl+n`: 下一条命令
- `ctrl+p`: 上一条命令
- `alt+n`: 下一条命令（例如输入`ls`, 然后按'alt+n', 就会找到历史记录下的`ls`命令）
- `alt+p`: 上一条命令（跟`alt+n`相似）
- `shift+PageUp`: 向上翻页
- `shift+PageDown`: 向下翻页
- `ctrl+r`: 进入历史查找命令记录， 输入关键字。 多次按返回下一个匹配项

## zsh

- `d`: 列出以前的打开的命令
- `j`: jump到以前某个目录，模糊匹配

## Vim

### 移动光标

- `b`: 向前移动一个单词
- `w`: 向后移动一个单词

### 删除

- `dw`: 从当前光标开始删除到下一个单词头
- `de`: 从当前光标开始删除到单词尾