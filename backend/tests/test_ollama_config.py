from backend.ollama import client as ollama_client


class FakeMemory:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path or "fake.csv"
        self.df = []
        self.data_summary = "fake summary"


def test_normalize_ollama_cloud_api_url_for_python_sdk():
    assert ollama_client._normalize_ollama_host("https://ollama.com/api") == "https://ollama.com"
    assert ollama_client._normalize_ollama_host("https://ollama.com/api/") == "https://ollama.com"
    assert ollama_client._normalize_ollama_host("http://ollama.com") == "https://ollama.com"
    assert ollama_client._normalize_ollama_host("http://ollama.com/api") == "https://ollama.com"


def test_primary_model_defaults_to_gemma_cloud(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    assert ollama_client._primary_model() == "gemma4:31b-cloud"


def test_cloud_host_is_preferred_for_cloud_model_with_api_key(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b-cloud")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434,https://ollama.com/api")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    hosts = ollama_client.ollama_host_candidates(model="gemma4:31b-cloud")

    assert hosts[0] == "https://ollama.com"
    assert "http://localhost:11434" in hosts


def test_cloud_host_is_added_for_cloud_model_with_api_key(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b-cloud")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    hosts = ollama_client.ollama_host_candidates(model="gemma4:31b-cloud")

    assert hosts[0] == "https://ollama.com"
    assert hosts[1] == "http://localhost:11434"


def test_cloud_host_is_not_added_without_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b-cloud")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    assert ollama_client.ollama_host_candidates(model="gemma4:31b-cloud") == ["http://localhost:11434"]


def test_ollama_client_uses_api_key_and_normalized_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

    client = ollama_client.OllamaClient(
        host="https://ollama.com/api",
        model="gemma4:31b-cloud",
        timeout_s=1,
    )

    assert client.host == "https://ollama.com"
    assert str(client.client._client.base_url).rstrip("/") == "https://ollama.com"
    assert client.client._client.headers.get("authorization") == "Bearer test-key"


def test_explicit_client_host_is_not_reordered_by_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b-cloud")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com/api")

    client = ollama_client.OllamaClient(host="http://localhost:11434", model="gemma4:31b-cloud")

    assert client.host == "http://localhost:11434"


def test_nfl_agent_uses_cloud_host_from_current_env(monkeypatch):
    from backend.ollama import llm_ollama

    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(llm_ollama, "NFLMemory", FakeMemory)

    agent = llm_ollama.NFLAgent(model="gemma4:31b-cloud")

    assert agent.host == "https://ollama.com"
    assert str(agent.client._client.base_url).rstrip("/") == "https://ollama.com"


def test_api_runtime_premium_agent_uses_cloud_aware_facade(monkeypatch):
    from backend.ollama import llm_ollama
    from backend.services import api_runtime

    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b-cloud,gemma4:e4b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.com/api,http://localhost:11434")
    monkeypatch.setattr(llm_ollama, "NFLMemory", FakeMemory)
    monkeypatch.setattr(api_runtime, "_nfl_agent", None)

    agent = api_runtime.get_nfl_agent()

    assert agent.model == "gemma4:31b-cloud"
    assert agent.host == "https://ollama.com"
    assert str(agent.client._client.base_url).rstrip("/") == "https://ollama.com"
