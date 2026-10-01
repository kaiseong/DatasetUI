"""One-use credentials, isolated from persistent registry and RQ Redis."""

from datasetui.config import Settings
from urllib.parse import urlsplit


def _redis_endpoint(url: str) -> tuple:
    address = urlsplit(url)
    if address.scheme == "unix":
        return ("unix", address.path)
    return (address.hostname, address.port or 6379)


class CredentialUnavailableError(RuntimeError):
    pass


class CredentialStore:
    def __init__(self, settings: Settings):
        if not settings.credential_redis_url or _redis_endpoint(
            settings.credential_redis_url
        ) == _redis_endpoint(settings.redis_url):
            raise CredentialUnavailableError(
                "임시 비밀번호 저장소가 설정되지 않았습니다."
            )
        from redis import Redis

        self.redis = Redis.from_url(
            settings.credential_redis_url, socket_timeout=3, socket_connect_timeout=3
        )
        self.ttl = settings.credential_ttl_seconds

    def put(self, job_id: str, password: str) -> None:
        # Refuse persistence even if an operator points this at the wrong instance.
        configuration = self.redis.config_get("save", "appendonly")
        if configuration.get(b"save", configuration.get("save")) not in {
            b"",
            "",
        } or configuration.get(b"appendonly", configuration.get("appendonly")) not in {
            b"no",
            "no",
        }:
            raise CredentialUnavailableError(
                "비밀번호 저장소의 디스크 저장을 비활성화해야 합니다."
            )
        self.redis.set(f"datasetui:credential:{job_id}", password, ex=self.ttl, nx=True)

    def take(self, job_id: str) -> str:
        value = self.redis.getdel(f"datasetui:credential:{job_id}")
        if value is None:
            raise CredentialUnavailableError(
                "비밀번호가 만료되거나 소실되었습니다. 비밀번호를 다시 입력해 새 작업을 요청하세요."
            )
        return value.decode() if isinstance(value, bytes) else value

    def exists(self, job_id: str) -> bool:
        return bool(self.redis.exists(f"datasetui:credential:{job_id}"))

    def delete(self, job_id: str) -> None:
        self.redis.delete(f"datasetui:credential:{job_id}")


def credential_store(settings: Settings) -> CredentialStore:
    return CredentialStore(settings)
