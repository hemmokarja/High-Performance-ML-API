import os
import random

import structlog
from dotenv import load_dotenv
from locust import HttpUser, task, events, constant_throughput

load_dotenv()
logger = structlog.get_logger(__name__)

API_KEY = os.getenv("API_KEY")


class EmbedUser(HttpUser):
    """
    Simulates users making single embedding requests.

    Throughput: 100 requests/second (adjust below)
    Pattern: Steady, constant rate
    """
    wait_time = constant_throughput(100)

    def on_start(self):
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise RuntimeError("API_KEY must be set as environment variable")

    @task
    def get_embedding(self):
        input_text = self._generate_input()
        payload = {"input_text": input_text}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        with self.client.post(
            "/v1/embed",
            json=payload,
            headers=headers,
            catch_response=True,
            name="/v1/embed"
        ) as response:
            try:
                if response.status_code != 200:
                    response.failure(f"Status {response.status_code}")
                    return

                data = response.json()
                if "embedding" not in data:
                    response.failure("Missing 'embedding' field")
                    return

                if not isinstance(data["embedding"], list):
                    response.failure("'embedding' is not a list")
                    return

                response.success()
                
            except ValueError as e:
                response.failure(f"Invalid JSON: {e}")
            except Exception as e:
                response.failure(f"Error: {e}")

    def _generate_input(self):
        """Generate varied input for testing"""
        patterns = [
            f"Query {random.randint(1, 100000)}",
            f"Document {random.randint(1, 10000)}",
            "Short text",
            "Longer text " * random.randint(5, 20),
        ]
        return random.choice(patterns)


@events.request.add_listener
def on_request_failure(
    request_type, name, response_time, response_length, exception, **kwargs
):
    if exception:
        logger.error(f"Request failed: {name} - {exception}")
