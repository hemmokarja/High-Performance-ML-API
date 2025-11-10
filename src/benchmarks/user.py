import time
import os
import random

import structlog
from dotenv import load_dotenv
from locust import HttpUser, task, events, constant_throughput

load_dotenv()
logger = structlog.get_logger(__name__)

class BaseEmbedUser(HttpUser):
    """Base class for embedding endpoint load tests"""

    abstract = True

    # override these in subclasses
    endpoint = "/v1/embed"
    requires_auth = True
    wait_time = constant_throughput(15)

    def on_start(self):
        if self.requires_auth:
            self.api_key = os.getenv("API_KEY")
            if not self.api_key:
                raise RuntimeError("API_KEY must be set as environment variable")
    
    @task
    def get_embedding(self):
        input_text = self._generate_input()
        payload = {"input_text": input_text}
        headers = self._get_headers()
        
        with self.client.post(
            self.endpoint,
            json=payload,
            headers=headers,
            catch_response=True,
            name=self.endpoint
        ) as response:
            self._validate_response(response)
    
    def _get_headers(self):
        """Build headers based on auth requirements"""
        headers = {"Content-Type": "application/json"}
        if self.requires_auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _validate_response(self, response):
        """Shared response validation logic"""
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
def on_request_failure(request_type, name, response_time, response_length, exception, **kwargs):
    if exception:
        logger.error(f"Request failed: {name} - {exception}")
