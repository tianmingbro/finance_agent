from pathlib import Path
fixtures = Path("tests/fixtures")
fixtures.mkdir(parents=True, exist_ok=True)

# TXT
with open(fixtures / "sample.txt", "w", encoding="utf-8") as f:
    f.write("《商业银行资本管理办法》：核心一级资本充足率不得低于5%。")

# PDF 
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()

# 【修改点1】直接使用 Windows 系统自带的微软雅黑字体绝对路径
# 【修改点2】新版 fpdf2 已默认支持 Unicode，直接去掉 uni=True 参数消除警告
pdf.add_font('ChineseFont', fname='C:/Windows/Fonts/msyh.ttc') 
pdf.set_font("ChineseFont", size=12) 

pdf.multi_cell(0, 10, "《存款保险条例》第五条：存款保险实行限额偿付，最高偿付限额为人民币50万元。")
pdf.output(str(fixtures / "sample.pdf"))

# DOCX 
from docx import Document
doc = Document()
doc.add_paragraph("《个人外汇管理办法》规定：个人每年结汇和购汇的便利化额度为等值5万美元。")
doc.save(str(fixtures / "sample.docx"))

print("测试夹具文件生成成功！")