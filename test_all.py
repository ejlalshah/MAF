"""
FRIDAY-MAF Test Suite
Unit tests covering: models, state machine, event bus, memory, tool router,
critics, workers, and the full pipeline (mocked LLM).

Run with:
    pytest tests/ -v
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# ── Path fix for running from project root ──────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# ────────────────────────────────────────────────────────────────────────────

from core.models import (
    AgentType, CriticResult, CriticType, Event, EventType,
    SubTask, Task, TaskStatus, WorkerOutput,
)
from core.state.state_machine import TaskStateMachine, StateMachineError
from core.events.event_bus import EventBus
from core.memory.memory_manager import MemoryManager, ShortTermMemory, EpisodicMemory, SemanticMemory
from core.tools.tool_router import ToolRouter, FileTool, WebSearchTool, PythonExecutionTool, AGENT_PERMISSIONS


# ===========================================================================
# Helpers
# ===========================================================================

def make_llm(response: str = '{"goal": "test goal", "metadata": {}}') -> MagicMock:
    """Return a mock LLM provider that always returns `response`."""
    llm = MagicMock()
    from core.models import LLMResponse
    llm.generate = AsyncMock(return_value=LLMResponse(content=response, model="mock"))
    llm.embed    = AsyncMock(return_value=[0.1] * 10)
    return llm


def make_memory() -> MemoryManager:
    return MemoryManager()


def make_event_bus() -> EventBus:
    return EventBus()


def make_subtask(agent: AgentType = AgentType.RESEARCH) -> SubTask:
    return SubTask(
        task_description = "Research the latest AI trends",
        required_agent   = agent,
    )


# ===========================================================================
# 1. Models
# ===========================================================================

class TestModels:
    def test_task_creates_with_uuid(self):
        t = Task(user_input="hello", goal="do something")
        assert len(t.task_id) > 0

    def test_subtask_defaults(self):
        st = make_subtask()
        assert st.status == TaskStatus.CREATED
        assert st.retry_count == 0
        assert st.dependencies == []

    def test_worker_output_ok(self):
        wo = WorkerOutput(agent_type=AgentType.CODING, task_id="x", success=True, result={"code": "print(1)"})
        assert wo.success is True

    def test_critic_result_fields(self):
        cr = CriticResult(critic_type=CriticType.LOGIC, passed=True, score=85, feedback="good")
        assert cr.score == 85

    def test_event_has_timestamp(self):
        e = Event(event_type=EventType.TASK_CREATED, task_id="abc")
        assert e.timestamp is not None


# ===========================================================================
# 2. State Machine
# ===========================================================================

class TestStateMachine:
    def test_valid_transitions(self):
        sm = TaskStateMachine("t1")
        sm.transition(TaskStatus.PLANNING)
        sm.transition(TaskStatus.EXECUTING)
        sm.transition(TaskStatus.REVIEWING)
        sm.transition(TaskStatus.COMPLETED)
        assert sm.state == TaskStatus.COMPLETED

    def test_invalid_transition_raises(self):
        sm = TaskStateMachine("t2")
        with pytest.raises(StateMachineError):
            sm.transition(TaskStatus.COMPLETED)   # CREATED → COMPLETED is invalid

    def test_terminal_state_blocks_transitions(self):
        sm = TaskStateMachine("t3")
        sm.transition(TaskStatus.PLANNING)
        sm.transition(TaskStatus.FAILED)
        assert sm.is_terminal()
        with pytest.raises(StateMachineError):
            sm.transition(TaskStatus.EXECUTING)

    def test_retry_limit_forces_failed(self):
        sm = TaskStateMachine("t4")
        sm.transition(TaskStatus.PLANNING)
        sm.transition(TaskStatus.EXECUTING)
        sm.transition(TaskStatus.RETRYING)
        sm.transition(TaskStatus.EXECUTING)
        sm.transition(TaskStatus.RETRYING)
        sm.transition(TaskStatus.EXECUTING)
        # 3rd retry hits MAX_RETRIES (>=3) → forced FAILED instead of RETRYING
        sm.transition(TaskStatus.RETRYING)
        assert sm.state == TaskStatus.FAILED

    def test_history_recorded(self):
        sm = TaskStateMachine("t5")
        sm.transition(TaskStatus.PLANNING)
        assert len(sm.history()) == 1
        assert sm.history()[0] == (TaskStatus.CREATED, TaskStatus.PLANNING)


# ===========================================================================
# 3. Event Bus
# ===========================================================================

class TestEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        bus     = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.TASK_CREATED, handler)
        event = Event(event_type=EventType.TASK_CREATED, task_id="t1")
        await bus.publish(event)
        assert len(received) == 1
        assert received[0].task_id == "t1"

    @pytest.mark.asyncio
    async def test_failing_handler_doesnt_block_others(self):
        bus     = EventBus()
        results = []

        async def bad_handler(e):
            raise RuntimeError("boom")

        async def good_handler(e):
            results.append("ok")

        bus.subscribe(EventType.TASK_CREATED, bad_handler)
        bus.subscribe(EventType.TASK_CREATED, good_handler)
        await bus.publish(Event(event_type=EventType.TASK_CREATED, task_id="t"))
        assert results == ["ok"]

    @pytest.mark.asyncio
    async def test_unsubscribed_event_type_is_noop(self):
        bus = EventBus()
        # Should not raise
        await bus.publish(Event(event_type=EventType.PLAN_COMPLETED, task_id="t"))

    @pytest.mark.asyncio
    async def test_history_is_recorded(self):
        # Python 3.12 removed the implicit default event loop in sync context;
        # making this async is the correct modern pattern.
        bus   = EventBus()
        event = Event(event_type=EventType.TASK_FAILED, task_id="t")
        await bus.publish(event)
        assert len(bus.history()) == 1


# ===========================================================================
# 4. Memory
# ===========================================================================

class TestShortTermMemory:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        mem = ShortTermMemory()
        await mem.store("key1", {"a": 1})
        val = await mem.retrieve("key1")
        assert val == {"a": 1}

    @pytest.mark.asyncio
    async def test_retrieve_missing_returns_none(self):
        mem = ShortTermMemory()
        assert await mem.retrieve("nonexistent") is None

    @pytest.mark.asyncio
    async def test_delete(self):
        mem = ShortTermMemory()
        await mem.store("k", "v")
        deleted = await mem.delete("k")
        assert deleted is True
        assert await mem.retrieve("k") is None

    @pytest.mark.asyncio
    async def test_update(self):
        mem = ShortTermMemory()
        await mem.store("k", 1)
        await mem.update("k", 2)
        assert await mem.retrieve("k") == 2


class TestEpisodicMemory:
    @pytest.mark.asyncio
    async def test_store_and_search(self):
        mem = EpisodicMemory()
        await mem.store("ep1", "The quick brown fox")
        results = await mem.search("fox")
        assert len(results) > 0
        assert results[0]["key"] == "ep1"

    @pytest.mark.asyncio
    async def test_empty_search_returns_empty(self):
        mem     = EpisodicMemory()
        results = await mem.search("anything")
        assert results == []


class TestSemanticMemory:
    @pytest.mark.asyncio
    async def test_store_and_lookup(self):
        mem = SemanticMemory()
        await mem.store("python_docs", "Python is a high-level programming language")
        results = await mem.search("programming language")
        assert len(results) > 0


# ===========================================================================
# 5. Tool Router
# ===========================================================================

class TestToolRouter:
    @pytest.mark.asyncio
    async def test_permission_denied(self):
        router = ToolRouter()
        router.register(WebSearchTool())
        result = await router.call("web_search", AgentType.CODING)  # coding can't search
        assert result["success"] is False
        assert "not permitted" in result["error"]

    @pytest.mark.asyncio
    async def test_permission_granted(self):
        router = ToolRouter()
        router.register(WebSearchTool())
        result = await router.call("web_search", AgentType.RESEARCH, query="test")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_unregistered_tool(self):
        router = ToolRouter()
        result = await router.call("nonexistent_tool", AgentType.RESEARCH)
        assert result["success"] is False

    def test_list_tools_for_agent(self):
        router = ToolRouter()
        router.register(WebSearchTool())
        router.register(FileTool())
        tools = router.list_tools(AgentType.RESEARCH)
        assert "web_search" in tools
        assert "python_exec" not in tools   # research can't exec

    @pytest.mark.asyncio
    async def test_file_tool_read_write(self, tmp_path):
        tool = FileTool()
        path = str(tmp_path / "test.txt")
        await tool.run(operation="write", path=path, content="hello")
        result = await tool.run(operation="read", path=path)
        assert result["data"] == "hello"

    @pytest.mark.asyncio
    async def test_python_exec_tool(self):
        tool   = PythonExecutionTool()
        result = await tool.run(code="print('hello world')")
        assert result["success"] is True
        assert "hello world" in result["data"]

    @pytest.mark.asyncio
    async def test_python_exec_captures_error(self):
        tool   = PythonExecutionTool()
        result = await tool.run(code="raise ValueError('intentional')")
        assert result["success"] is False


# ===========================================================================
# 6. Critics
# ===========================================================================

class TestCritics:
    def _make_critic(self, critic_class, llm_response: dict):
        from core.critics.critics import LogicCritic, SecurityCritic, RequirementCritic
        llm    = make_llm(json.dumps(llm_response))
        memory = make_memory()
        bus    = make_event_bus()
        return critic_class(llm, memory, bus)

    @pytest.mark.asyncio
    async def test_logic_critic_pass(self):
        from core.critics.critics import LogicCritic
        critic  = self._make_critic(LogicCritic, {"pass": True, "score": 90, "feedback": "great"})
        subtask = make_subtask()
        output  = WorkerOutput(agent_type=AgentType.RESEARCH, task_id=subtask.task_id, success=True, result="some result")
        result  = await critic.run(subtask, output)
        assert result.passed is True
        assert result.score == 90

    @pytest.mark.asyncio
    async def test_security_critic_fail(self):
        from core.critics.critics import SecurityCritic
        critic  = self._make_critic(SecurityCritic, {"pass": False, "score": 20, "feedback": "SQL injection found"})
        subtask = make_subtask()
        output  = WorkerOutput(agent_type=AgentType.CODING, task_id=subtask.task_id, success=True, result="bad code")
        result  = await critic.run(subtask, output)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_critic_runner_all_pass(self):
        from core.critics.critics import CriticRunner, LogicCritic, SecurityCritic, RequirementCritic
        critics = [
            self._make_critic(LogicCritic,       {"pass": True, "score": 80, "feedback": "ok"}),
            self._make_critic(SecurityCritic,    {"pass": True, "score": 90, "feedback": "ok"}),
            self._make_critic(RequirementCritic, {"pass": True, "score": 75, "feedback": "ok"}),
        ]
        runner  = CriticRunner(critics)
        subtask = make_subtask()
        output  = WorkerOutput(agent_type=AgentType.RESEARCH, task_id=subtask.task_id, success=True, result="r")
        passed, results = await runner.evaluate(subtask, output)
        assert passed is True

    @pytest.mark.asyncio
    async def test_critic_runner_one_fail_means_overall_fail(self):
        from core.critics.critics import CriticRunner, LogicCritic, SecurityCritic
        critics = [
            self._make_critic(LogicCritic,    {"pass": True,  "score": 80, "feedback": "ok"}),
            self._make_critic(SecurityCritic, {"pass": False, "score": 30, "feedback": "vuln"}),
        ]
        runner  = CriticRunner(critics)
        subtask = make_subtask()
        output  = WorkerOutput(agent_type=AgentType.RESEARCH, task_id=subtask.task_id, success=True, result="r")
        passed, _ = await runner.evaluate(subtask, output)
        assert passed is False


# ===========================================================================
# 7. Communicator Agent
# ===========================================================================

class TestCommunicatorAgent:
    @pytest.mark.asyncio
    async def test_run_returns_task(self):
        from core.agents.communicator import CommunicatorAgent
        llm_resp = json.dumps({"goal": "find AI papers", "metadata": {"domain": "research"}})
        agent    = CommunicatorAgent(make_llm(llm_resp), make_memory(), make_event_bus())
        task     = await agent.run("Find recent AI research papers")
        assert isinstance(task, Task)
        assert "AI" in task.goal or task.goal == "find AI papers"

    @pytest.mark.asyncio
    async def test_format_response_completed(self):
        from core.agents.communicator import CommunicatorAgent
        agent = CommunicatorAgent(make_llm("Here are your results."), make_memory(), make_event_bus())
        task  = Task(user_input="hello", goal="test", status=TaskStatus.COMPLETED, final_result="done")
        resp  = await agent.format_response(task)
        assert isinstance(resp, str)

    @pytest.mark.asyncio
    async def test_format_response_failed(self):
        from core.agents.communicator import CommunicatorAgent
        agent = CommunicatorAgent(make_llm(""), make_memory(), make_event_bus())
        task  = Task(user_input="hello", goal="test", status=TaskStatus.FAILED, error="network error")
        resp  = await agent.format_response(task)
        assert "sorry" in resp.lower() or "network error" in resp.lower()


# ===========================================================================
# 8. Memory Manager (Integration)
# ===========================================================================

class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_task_state(self):
        mgr  = MemoryManager()
        task = Task(user_input="hi", goal="do stuff")
        # model_dump(mode='json') serializes datetime → ISO string (Pydantic v2)
        await mgr.store_task_state(task.task_id, task.model_dump(mode="json"))
        result = await mgr.retrieve_task_state(task.task_id)
        assert result is not None
        assert result["task_id"] == task.task_id

    @pytest.mark.asyncio
    async def test_record_and_recall_episode(self):
        mgr = MemoryManager()
        await mgr.record_episode("task1", "We researched quantum computing")
        results = await mgr.recall_similar("quantum")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_learn_and_lookup(self):
        mgr = MemoryManager()
        await mgr.learn("python_basics", "Python uses indentation for blocks")
        results = await mgr.lookup("indentation")
        assert len(results) > 0


# ===========================================================================
# 9. Full Pipeline (Mocked LLM)
# ===========================================================================

class TestPipeline:
    """End-to-end smoke test using a mocked LLM."""

    def _plan_response(self) -> str:
        return json.dumps([
            {"task_description": "Search for AI trends", "required_agent": "research", "dependencies": []},
        ])

    @pytest.mark.asyncio
    async def test_plan_creates_subtasks(self):
        from core.agents.planner import PlannerAgent
        llm   = make_llm(self._plan_response())
        agent = PlannerAgent(llm, make_memory(), make_event_bus())
        task  = Task(user_input="AI trends", goal="Research the latest AI trends")
        task  = await agent.run(task)
        assert len(task.subtasks) == 1
        assert task.subtasks[0].required_agent == AgentType.RESEARCH

    @pytest.mark.asyncio
    async def test_planner_detects_circular_dependency(self):
        from core.agents.planner import PlannerAgent
        agent = PlannerAgent(make_llm("[]"), make_memory(), make_event_bus())
        # Manually construct circular deps
        st_a  = SubTask(task_description="A", required_agent=AgentType.RESEARCH)
        st_b  = SubTask(task_description="B", required_agent=AgentType.RESEARCH, dependencies=[st_a.task_id])
        st_a.dependencies = [st_b.task_id]   # circular!
        with pytest.raises(ValueError, match="Circular"):
            agent._execution_order([st_a, st_b])