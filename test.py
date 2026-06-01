from src.agent.agent import build_agent
agent = build_agent()
print(agent.get_graph().draw_ascii())  # 查看图结构