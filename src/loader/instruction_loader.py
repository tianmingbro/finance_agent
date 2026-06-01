"""
指令加载器 - 金融 RAG Skill 第二层
职责：
1. 维护可配置的系统指令（默认行为 + 护栏规则）
2. 基于触发词判断是否加载指令
3. 将指令注入 LLM 上下文模板
"""
import re
from typing import List, Optional, Dict

# -------------------- 可配置指令集 --------------------
FINANCE_INSTRUCTIONS: Dict[str, str] = {
    "default_behavior": (
        "你是一个专业的金融法规咨询助手。"
        "使用以下检索到的上下文来回答问题。"
        "回答请保持简洁、专业，不超过三句话。"
    ),
    "safety_guard": (
        "如果上下文中找不到答案，请诚实地说'该信息未在已知法规中收录'。"
        "如果用户问题涉及违法或不当请求，请明确拒绝。"
        "不要编造任何法规条款或数字。"
    ),
    "clarification": (
        "如果用户问题不明确或存在歧义，请主动询问澄清细节，而不是猜测。"
    ),
}

# -------------------- 触发词清单 --------------------
TRIGGER_KEYWORDS: List[str] = [
    "资本充足率", "LPR", "贷款市场报价利率", "购汇额度", "外汇管理",
    "金融法规", "监管要求", "商业银行", "房贷利率", "外汇额度",
    "资本管理办法", "核心一级资本", "便利化额度"
]


class InstructionLoader:
    """渐进式加载的第二层：指令管理"""

    def __init__(self, trigger_keywords: List[str] = None,
                 instructions: Dict[str, str] = None):
        self.trigger_keywords = trigger_keywords or TRIGGER_KEYWORDS
        self.instructions = instructions or FINANCE_INSTRUCTIONS

        # 编译关键词正则，忽略大小写
        escaped = [re.escape(kw) for kw in self.trigger_keywords]
        self.trigger_pattern = re.compile("|".join(escaped), re.IGNORECASE)

    def should_trigger(self, user_input: str) -> bool:
        """判断用户输入是否匹配任意触发词"""
        return bool(self.trigger_pattern.search(user_input))

    def load_instruction(self, user_input: str = None,
                         include_safety: bool = True,
                         include_clarification: bool = True) -> str:
        """
        加载系统指令，可拼接成完整 Prompt 模板。
        如果提供了 user_input，会自动判断是否应该触发；
        否则无条件返回指令（用于测试）。
        """
        if user_input is not None and not self.should_trigger(user_input):
            return ""  # 不触发则返回空指令

        parts = [self.instructions["default_behavior"]]
        if include_safety:
            parts.append(self.instructions["safety_guard"])
        if include_clarification:
            parts.append(self.instructions["clarification"])

        return "\n".join(parts)

    def inject_into_prompt(self, base_prompt: str,
                          user_input: str,
                          include_safety: bool = True,
                          include_clarification: bool = True) -> str:
        """
        将指令注入到基础 Prompt 模板中。
        基础模板应包含 {instruction} 和 {input} 占位符。
        """
        instruction = self.load_instruction(
            user_input=user_input,
            include_safety=include_safety,
            include_clarification=include_clarification
        )
        return base_prompt.format(instruction=instruction, input=user_input)


# -------------------- 测试脚本 --------------------
def test_instruction_loader():
    """自动化测试：指令加载与注入的正确性"""
    loader = InstructionLoader()

    # 测试 1：触发词匹配
    assert loader.should_trigger("LPR最新报价是多少？"), "触发词 LPR 应被匹配"
    assert not loader.should_trigger("今天天气怎么样？"), "无触发词应返回 False"

    # 测试 2：指令加载包含关键内容
    full_instruction = loader.load_instruction()
    assert "金融法规咨询助手" in full_instruction, "应包含默认行为描述"
    assert "该信息未在已知法规中收录" in full_instruction, "应包含安全护栏"
    assert "主动询问澄清" in full_instruction, "应包含澄清规则"

    # 测试 3：不触发时返回空指令
    no_trigger_instruction = loader.load_instruction(user_input="天气如何")
    assert no_trigger_instruction == "", "不触发时应返回空字符串"

    # 测试 4：指令注入到 Prompt 模板
    base_prompt = "系统指令:\n{instruction}\n\n用户问题: {input}\n回答:"
    result = loader.inject_into_prompt(
        base_prompt, "资本充足率是多少？"
    )
    assert "金融法规咨询助手" in result, "注入后应包含指令"
    assert "资本充足率是多少？" in result, "注入后应包含用户问题"

    print("✅ 所有测试通过！")


if __name__ == "__main__":
    test_instruction_loader()