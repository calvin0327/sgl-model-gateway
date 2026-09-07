"""Basic CRUD tests for the Response API against cloud backends.

Source: Migrated from e2e_response_api/features/test_basic_crud.py
"""

from __future__ import annotations

import logging
import time

import openai
import pytest
from openai import OpenAI
from openai.types import responses

logger = logging.getLogger(__name__)


def wait_for_background_task(
    client: OpenAI, response_id: str, timeout: int = 30, poll_interval: float = 0.5
) -> responses.Response:
    """Wait for background task to complete.

    Args:
        client: OpenAI client
        response_id: Response ID to poll
        timeout: Max seconds to wait
        poll_interval: Seconds between polls

    Returns:
        Final response data

    Raises:
        TimeoutError: If task doesn't complete in time
        AssertionError: If task fails
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        resp = client.responses.retrieve(response_id=response_id)
        assert resp.error is None
        assert resp.id == response_id

        if resp.status == "completed":
            return resp
        elif resp.status == "failed":
            raise AssertionError(f"Background task failed: {resp.error}")
        elif resp.status == "cancelled":
            raise AssertionError("Background task was cancelled")

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Background task {response_id} did not complete within {timeout}s"
    )


# =============================================================================
# Response CRUD Tests (Memory Storage)
# =============================================================================


@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestResponseCRUD:
    """Tests for Response API CRUD operations."""

    def test_create_and_get_response(self, setup_backend):
        """Test creating response and retrieving it."""
        _, model, client, gateway = setup_backend

        # Create response
        create_resp = client.responses.create(model=model, input="Hello, world!")
        assert create_resp.id is not None
        assert create_resp.error is None
        assert create_resp.status == "completed"
        assert len(create_resp.output_text) > 0
        response_id = create_resp.id

        # Get response
        get_resp = client.responses.retrieve(response_id=response_id)
        assert get_resp.error is None
        assert get_resp.id == response_id
        assert get_resp.status == "completed"

        input_resp = client.responses.input_items.list(response_id=get_resp.id)
        assert input_resp.data is not None
        assert len(input_resp.data) > 0

    @pytest.mark.skip(reason="TODO: Add delete response feature")
    def test_delete_response(self, setup_backend):
        """Test deleting response."""
        _, model, client, gateway = setup_backend

        # Create response
        create_resp = client.responses.create(model=model, input="Test deletion")
        assert create_resp.id is not None
        assert create_resp.error is None
        assert create_resp.status == "completed"
        assert len(create_resp.output_text) > 0

        response_id = create_resp.id

        # Delete response
        client.responses.delete(response_id=response_id)

        # Verify it's deleted (should return 404)
        with pytest.raises(openai.NotFoundError):
            client.responses.retrieve(response_id=response_id)

    @pytest.mark.skip(reason="TODO: Add background response feature")
    def test_background_response(self, setup_backend):
        """Test background response execution."""
        _, model, client, gateway = setup_backend

        # Create background response
        create_resp = client.responses.create(
            model=model,
            input="Write a short story",
            background=True,
            max_output_tokens=100,
        )
        assert create_resp.id is not None
        assert create_resp.error is None
        assert create_resp.status in ["in_progress", "queued"]

        response_id = create_resp.id

        # Wait for completion
        final_data = wait_for_background_task(client, response_id, timeout=60)
        assert final_data.status == "completed"
