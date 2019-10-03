from pathlib import Path
import os
import glob
work_dir = Path.cwd()

export_pdf_dir = work_dir / 'pdf2'
print(export_pdf_dir)
if not export_pdf_dir.exists():
    export_pdf_dir.mkdir()
#file=open('fix.md','w')  
##向文件中写入字符  
##file.write('test\n')  
#for md_file in list(sorted(glob.glob('./*/*.md'))):
#    for line in open(md_file):  
#        file.writelines(line)  
#    file.write('\n')
#pdf_file = "./Dive-into-DL-PyTorch.pdf"   
#cmd = "pandoc  -N --template=template2.tex --variable mainfont='PingFang SC' --variable sansfont='Helvetica' --variable monofont='Menlo' --variable fontsize=12pt --variable version=2.0 '{}' --latex-engine=xelatex --toc -o '{}' ".format("fix.md", pdf_file)
#os.system(cmd)
   
for md_file in list(sorted(glob.glob('./*/*.md'))):
    print(md_file)
    md_file_name = md_file
    zhanjie=md_file_name.split("/")[-2]
    print(zhanjie)
    pdf_file_name = md_file_name.replace('.md', '.pdf')
    pdf_file = export_pdf_dir/pdf_file_name
    os.makedirs(str(export_pdf_dir/zhanjie),exist_ok=True)
    print(pdf_file)
    cmd = "pandoc '{}' -o '{}' -s --highlight-style pygments  --latex-engine=xelatex -V mainfont='PingFang SC' --template=template.tex".format(md_file, pdf_file)
    os.system(cmd)