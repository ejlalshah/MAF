"""
FRIDAY-MAF Core Data Models
Shared Pydantic models and enums used across the entire framework.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from uuid import uuid4
from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


#----------------------------------------------------------------------------------------------------
# Enums


class TaskStatus(str, Enum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    REVIEWING = "REVIEWING"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AgentType(str, Enum):
    COMMUNICATOR = "communicator"
    PLANNER = "planner"
    ORCHESTRATOR = "orchestrator"
    RESEARCH = "research"
    CODING = "coding"
    DATA_ANALYST = "data_analyst"
    RECOVERY = "recovery"


class EventType(str, Enum):
    TASK_CREATED = "TASK_CREATED"
    TASK_ASSIGNED = "TASK_ASSIGNED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    CRITIC_FAILED = "CRITIC_FAILED"
    RETRY_TRIGGERED = "RETRY_TRIGGERED"
    PLAN_COMPLETED = "PLAN_COMPLETED"


class CriticType(str, Enum):
    LOGIC = "logic"
    SECURITY = "security"
    REQUIREMENT = "requirement"


#----------------------------------------------------------------------------------------------------
# Core Task Model
#----------------------------------------------------------------------------------------------------

class SubTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    task_description: str
    required_agent: AgentType
    dependencies: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.CREATED
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def mark_updated(self) -> None:
        self.updated_at = _now()


class Task(BaseModel):
    task_id:       str           = Field(default_factory=lambda: str(uuid4()))
    user_input:    str
    goal:          str
    status:        TaskStatus    = TaskStatus.CREATED
    subtasks:      List[SubTask] = Field(default_factory=list)
    final_result:  Optional[Any] = None
    error:         Optional[str] = None
    metadata:      Dict[str, Any] = Field(default_factory=dict)
    created_at:    datetime       = Field(default_factory=_now)
    updated_at:    datetime       = Field(default_factory=_now)

    def mark_updated(self) -> None:
        self.updated_at = _now()


#----------------------------------------------------------------------------------------------------
# Critic Output
#----------------------------------------------------------------------------------------------------

class CriticResult(BaseModel):
    critic_type: CriticType
    passed: bool
    score: int = Field(ge=0, le=100)
    feedback: str


#----------------------------------------------------------------------------------------------------
# Worker Output
#----------------------------------------------------------------------------------------------------

class WorkerOutput(BaseModel):
    agent_type: AgentType
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


#----------------------------------------------------------------------------------------------------
# Event Model
#----------------------------------------------------------------------------------------------------

class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    task_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)


#----------------------------------------------------------------------------------------------------
# LLM Message
#----------------------------------------------------------------------------------------------------

class LLMMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "Stop"
