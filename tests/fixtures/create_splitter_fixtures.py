# tests/fixtures/create_splitter_fixtures.py
from pathlib import Path

FIXTURES = Path(__file__).parent

# 纯文本文档（多段落，适合 recursive 和 semantic）
plain_text = """《商业银行资本管理办法》于2024年1月1日起施行。

核心一级资本充足率不得低于5%。
一级资本充足率不得低于6%。
资本充足率不得低于8%。"""

with open(FIXTURES / "plain.txt", "w", encoding="utf-8") as f:
    f.write(plain_text)

# Markdown 文档（带标题层级，适合 markdown 策略）
markdown_text = """# 存款保险条例
## 第一条 立法目的
为了建立和规范存款保险制度，依法保护存款人的合法权益，及时防范和化解金融风险，维护金融稳定，制定本条例。

## 第二条 投保机构
在中华人民共和国境内设立的商业银行、农村合作银行、农村信用合作社等吸收存款的银行业金融机构，应当依照本条例的规定投保存款保险。

### 第三款 适用范围
其他经国务院银行业监督管理机构批准设立的金融机构，适用本条例。"""

with open(FIXTURES / "regulation.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)

# 多段落文本（长文，验证重叠和 chunk_size 限制）
long_text = "\n".join([
    f"第{i}条 本法规定的内容包括但不限于上述条款，具体实施由相关监管部门负责。" 
    for i in range(1, 20)
])
with open(FIXTURES / "long_text.txt", "w", encoding="utf-8") as f:
    f.write(long_text)

print("✅ 测试夹具已生成。")