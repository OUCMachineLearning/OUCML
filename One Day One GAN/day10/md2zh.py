
import re
def repl(m):
	inner_word = m.group(1)
	return '<br><br>$' + inner_word + '$<br><br>'
with open('readme.md', 'r') as f_read:
	text = f_read.readlines()
	for k, item in enumerate(text):
		text[k] = re.sub(r'\$\$(.*?)\$\$', repl, item)
	
	with open('zhihu.md', 'w') as f_write:
		f_write.writelines(text)