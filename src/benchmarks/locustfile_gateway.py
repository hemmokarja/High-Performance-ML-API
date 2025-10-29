from benchmarks.user import BaseEmbedUser

class GatewayEmbedUser(BaseEmbedUser):
    endpoint = "/v1/embed"
    requires_auth = True
