# Deploying the API on a Windows VM (Git + NSSM)

How the service gets from a Git repository onto a Windows VM and runs there as a Windows service
managed by **NSSM** (the Non-Sucking Service Manager). Placeholders are written as `<...>` - fill them in
for your own setup. Never write real keys, passwords or connection strings into this file.

| Placeholder | Meaning |
| --- | --- |
| `<repo-url>` | The Git repository's HTTPS or SSH URL |
| `<service-folder>` | Folder on the VM the code is cloned into, e.g. `D:\apps\ocr-api` |
| `<ServiceName>` | Windows service name (no spaces), e.g. `OcrApi` |
| `<Display Name>` | Name shown in `services.msc` - must be unique on the VM |
| `<port>` | Port the API listens on |

---

## 1. How it fits together

```
Developer PC                      Git host                        Windows VM
------------                      --------                        ----------
working copy  --(tests pass)-->   push to main  --(git pull)-->   <service-folder>
                                                                    .venv  (Python packages)
                                                                    .env   (secrets, VM only)
                                                                    logs\
                                                                  NSSM service <ServiceName>
                                                                    runs: .venv\Scripts\python.exe -m app
                                                                    starts at boot, restarts on crash
```

- **Git carries only code.** Secrets (`.env`), the virtual environment, logs and output files never go into
  Git - they are listed in `.gitignore` and exist only on the machine that uses them.
- **NSSM** turns `python -m app` (a normal console program) into a real Windows service: it starts at
  boot, restarts after a crash, and writes the console output to log files.

---

## 2. One-time VM setup

Run everything in an **Administrator PowerShell**.

### 2.1 Install the tools
```powershell
winget install Python.Python.3.12
winget install Git.Git
winget install NSSM.NSSM          # or: choco install nssm
```
Open a new PowerShell afterwards so `py`, `git` and `nssm` are on the PATH, then check:
```powershell
py -3.12 --version; git --version; nssm version
```

### 2.2 Give the VM read-only access to the repository
Use a credential that can **read** the repository and nothing else:
- **SSH deploy key** (preferred): `ssh-keygen -t ed25519 -f $HOME\.ssh\deploy_key`, add the `.pub` file to
  the repository as a read-only deploy key, and clone with the SSH URL.
- **or a fine-grained access token** limited to this one repository with read-only "Contents" permission.
  Git Credential Manager stores it on first clone.

Never use a personal account password, and never give the VM write access.

### 2.3 Clone the code
```powershell
git clone <repo-url> <service-folder>
cd <service-folder>
```

### 2.4 Create the `.env` (secrets live only here)
```powershell
Copy-Item .env.example .env
notepad .env
```
Fill in at least: `APP_HOST=0.0.0.0`, `APP_PORT=<port>`, the Gemini key and model, the blob storage
settings and `API_KEYS` (the keys callers send in `X-API-Key`). Then lock the file down so only
Administrators and the service account can read it:
```powershell
icacls .env /inheritance:r /grant:r "Administrators:F" "SYSTEM:R"
```
Generate API keys with a random generator, never by hand, for example:
```powershell
python -c "import secrets; print('ocr_' + secrets.token_urlsafe(32))"
```

### 2.5 Register the service with NSSM
The install script creates `.venv`, installs `requirements.txt` and registers the service:
```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows\install-service.ps1 `
    -ServiceName <ServiceName> -DisplayName "<Display Name>"
```
What it sets in NSSM:

| Setting | Value | Why |
| --- | --- | --- |
| Application | `.venv\Scripts\python.exe -m app` | runs the API |
| AppDirectory | `<service-folder>` | so `.env` and `logs\` are found |
| Start | Automatic | starts at boot |
| AppExit / AppRestartDelay / AppThrottle | Restart, 5 s, 10 s | restarts after a crash without spinning the CPU |
| AppStopMethodConsole | 15 s | lets in-flight requests finish on stop |
| AppStdout / AppStderr | `logs\service-stdout.log`, `logs\service-stderr.log` | console output, rotated at 10 MB |
| AppEnvironmentExtra | `PYTHONUNBUFFERED=1` | log lines appear immediately |

**Each service on the VM needs its own service name and display name.** NSSM can resolve a service
through its display name, so two services with the same display name make every `nssm` command act on the
wrong one. The script refuses a display name that is already taken. To fix an existing clash:
```powershell
sc.exe config <ServiceName> DisplayName= "<Display Name>"
```

### 2.6 Open the port
```powershell
New-NetFirewallRule -DisplayName "<Display Name> <port>" -Direction Inbound -Protocol TCP `
    -LocalPort <port> -Action Allow -RemoteAddress <allowed caller IPs>
```
Limit `-RemoteAddress` to the servers that call the API (see section 5).

### 2.7 Check it is running
```powershell
nssm status <ServiceName>                                   # SERVICE_RUNNING
Invoke-RestMethod http://127.0.0.1:<port>/health            # on the VM
```
From another machine: `http://<vm-address>:<port>/health`.

---

## 3. Deploying an update

1. On the PC: make the change, run the tests, test locally.
2. Before pushing, check nothing secret or runtime is staged:
   ```powershell
   git status --short          # no .env, logs/, .venv/, output folders
   git diff --cached | Select-String -Pattern "AIza|AccountKey=|DefaultEndpointsProtocol|sig=|PRIVATE KEY"
   ```
   The second command must print nothing.
3. Push to `main`.
4. On the VM (Administrator PowerShell):
   ```powershell
   cd <service-folder>
   powershell -ExecutionPolicy Bypass -File deploy\windows\update-service.ps1 -ServiceName <ServiceName>
   ```
   The script runs `git pull --ff-only`, reinstalls requirements and restarts the service through NSSM.
5. Wait until the API answers (startup takes a few seconds - "Unable to connect" right after the restart
   is normal), then confirm the deployed commit:
   ```powershell
   foreach ($i in 1..30) { try { Invoke-RestMethod http://127.0.0.1:<port>/health -TimeoutSec 2 | Out-Null; "up"; break } catch { Start-Sleep 2 } }
   git log -1 --oneline
   ```

**Rollback:** `git checkout <previous-commit>` in `<service-folder>`, then `nssm restart <ServiceName>`.
Return to `main` with `git checkout main` before the next update.

`.env` is not touched by an update. When a new setting is added to `.env.example`, add it to the VM's
`.env` by hand before restarting.

---

## 4. Everyday NSSM commands

```powershell
nssm status  <ServiceName>
nssm start   <ServiceName>
nssm stop    <ServiceName>
nssm restart <ServiceName>
nssm edit    <ServiceName>      # opens the settings window
nssm get     <ServiceName> AppDirectory
powershell -ExecutionPolicy Bypass -File deploy\windows\uninstall-service.ps1 -ServiceName <ServiceName>
```

Logs are in `<service-folder>\logs\`: `ocr-api.log` (what the API did), `errors.log` (warnings and
errors), `usage.log` (tokens per request), and `service-stdout.log` / `service-stderr.log` (start-up
errors - look here first when the service will not stay running).

| Problem | Check |
| --- | --- |
| Service stops right after starting | `logs\service-stderr.log` - usually a missing or wrong `.env` value |
| `nssm` acts on the wrong service | two services share a display name - see 2.5 |
| Works on the VM, not from outside | firewall rule, `APP_HOST=0.0.0.0`, cloud provider's network rules |
| `git pull` fails | local edits on the VM - code is only ever changed through Git, never on the VM |

---

## 5. Security checklist

- **Secrets only in `.env` on the VM**, file permissions restricted (2.4). Never in Git, never in logs,
  never in chat or tickets. If a key is ever exposed, rotate it at once.
- **Read-only repository access** from the VM (2.2); keep the repository private.
- **API keys required:** `API_KEYS` must not be empty in production; give each caller its own key so one
  can be revoked alone.
- **HTTPS in front of the API** before go-live: a reverse proxy (IIS with URL Rewrite/ARR, or Caddy/nginx)
  holding the certificate, with the API itself listening only on `127.0.0.1` and `REQUIRE_HTTPS=true`.
- **API docs off in production:** `DOCS_ENABLED=false`.
- **Firewall:** allow the API port only from the calling servers' addresses, never from everywhere.
- **Personal data:** keep `LOG_EXTRACTED_RESULTS=false`; processed files are deleted after
  `OUTPUT_RETENTION_HOURS`.
- **Administrator access** to the VM only for the people who deploy; use the Administrator PowerShell
  only for install and update steps.
