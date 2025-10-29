from benchmarks.user import BaseEmbedUser

class InferenceEmbedUser(BaseEmbedUser):
    endpoint = "/embed"
    requires_auth = False
