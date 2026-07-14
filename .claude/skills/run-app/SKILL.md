---
name: run-app
description: Start the FastAPI backend and Vite frontend dev servers, then verify both are live
---

Launch both servers for the learning-while-doing project and confirm they respond.

## Steps

1. **Kill any stale processes** on ports 8000 and 5173:
   ```powershell
   Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -match 'uvicorn|vite' } | Stop-Process -Force
   ```
   Or use netstat to find and kill the PIDs:
   ```powershell
   $p8000 = (netstat -ano | Select-String ':8000').ToString().Trim().Split()[-1]
   if ($p8000) { Stop-Process -Id $p8000 -Force -ErrorAction SilentlyContinue }
   $p5173 = (netstat -ano | Select-String ':5173').ToString().Trim().Split()[-1]
   if ($p5173) { Stop-Process -Id $p5173 -Force -ErrorAction SilentlyContinue }
   ```

2. **Start the backend** (FastAPI on port 8000):
   ```powershell
   Start-Process -NoNewWindow powershell -ArgumentList '-Command', '.venv\Scripts\uvicorn.exe shell.app:app --reload --port 8000'
   ```

3. **Start the frontend** (Vite on port 5173):
   ```powershell
   Start-Process -NoNewWindow powershell -ArgumentList '-Command', 'cd frontend; npm run dev'
   ```

4. **Wait for both to respond** (poll every second, timeout 30s):
   ```powershell
   $timeout = 30; $i = 0
   while ($i -lt $timeout) {
     try { Invoke-RestMethod http://localhost:8000/health | Out-Null; break } catch {}
     Start-Sleep 1; $i++
   }
   if ($i -ge $timeout) { Write-Error "Backend did not start"; exit 1 }
   Write-Host "Backend live at http://localhost:8000"
   Write-Host "Frontend live at http://localhost:5173"
   ```

5. **Report status** — confirm both URLs are live and note any errors.

If Playwright MCP is available, take a screenshot of http://localhost:5173 and describe what's visible.
