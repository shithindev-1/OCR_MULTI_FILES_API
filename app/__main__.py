"""Production entry point: `python -m app`.

This is what the Windows service (NSSM) runs. Calling Uvicorn from Python rather than through
`uvicorn.exe` also sidesteps Windows Application Control policies that block unsigned launcher
executables inside a virtual environment.

One worker only: the rate limiter and per-request usage tracking are held in process memory.
"""

import uvicorn

from app.core.config import settings


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        # The app writes its own rotating files under logs/; Uvicorn's access log only goes to
        # stdout, which NSSM captures in logs/service-stdout.log.
        access_log=True,
    )


if __name__ == "__main__":
    main()
