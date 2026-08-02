import os
import pytest
from swarms.structs.agent import Agent


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="requires a live provider key"
)
def test_agent_stream_token():
    agent = Agent(
        model_name="gpt-5.4",
        max_loops=1,
        stream=True,
    )
    assert agent.run("Tell me a short story about a robot learning to paint.")
