"""A2A 协议核心数据结构（最小实现）—— 支持 JSON 序列化"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

@dataclass
class TextPart:
    text: str
    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text}

@dataclass
class Message:
    role: str = "user"          # "user" 或 "agent"
    parts: List[TextPart] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts]
        }

@dataclass
class AgentSkill:
    id: str
    name: str
    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name}

@dataclass
class AgentCard:
    name: str
    description: str
    url: str
    skills: List[AgentSkill] = field(default_factory=list)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "skills": [s.to_dict() for s in self.skills],
            "capabilities": self.capabilities,
        }

@dataclass
class Task:
    id: str
    session_id: str = ""
    messages: List[Message] = field(default_factory=list)
    artifacts: List[Any] = field(default_factory=list)
    status: str = "created"     # created / working / completed / failed
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "artifacts": self.artifacts,
            "status": self.status,
        }