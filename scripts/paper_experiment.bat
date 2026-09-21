@echo off
REM Launch paper-experiment detached from the parent shell.
REM
REM Motivating pattern: launching via Claude Code (or any interactive
REM harness that watches child processes for memory pressure) had the
REM supervisor reaped four times in one session on 2026-09-18 when the
REM host memory crossed ~89%. Each reap tore down the whole process
REM tree (arms A/B/C + ml-service + buses + dashboard) and cost ~$0.20
REM in cold-start LLM calls plus ~5 min of trading time to relaunch.
REM
REM This script uses `start` to spawn the experiment in a new console
REM window that's independent of whatever launched THIS batch, so the
REM experiment is decoupled from Claude Code's or any other shell's
REM lifecycle. Output goes to out\logs\experiment.out so you can
REM `tail -F` from any terminal.
REM
REM Usage from anywhere:
REM   scripts\paper_experiment.bat reasoning-quality
REM   scripts\paper_experiment.bat reasoning-quality --dashboard-port 8001
REM
REM Any args after the experiment name are passed through verbatim.
REM The default paper-loop-args (--auto --top-n 8 --regen-mode event)
REM match what was running for the 2026-09-14/18 A/B; override by
REM adding --paper-loop-args ... after the experiment name.

setlocal EnableDelayedExpansion
set REPO=%~dp0..
if "%~1"=="" (
  echo Usage: %~nx0 ^<experiment-name^> [extra paper-experiment args]
  echo Example: %~nx0 reasoning-quality
  exit /b 2
)
set EXP=%~1
shift

REM Collect any additional args (%2, %3, ...) into EXTRA. Windows
REM `shift` does NOT shift %*, so we can't reuse %*. Iterate through
REM the remaining positional args (`%~1` after the shift) and build
REM a space-joined string; if none, EXTRA stays empty. Without this
REM the previous version leaked the already-consumed experiment name
REM back onto the tail via %* and every arm's paper-loop process
REM crashed with `unrecognized arguments: <expname>`.
set EXTRA=
:collect_extras
if "%~1"=="" goto extras_done
set EXTRA=!EXTRA! %~1
shift
goto collect_extras
:extras_done

REM Belt + suspenders: even though the new console is independent of
REM this shell's parent, set the harness env var too so a downstream
REM Claude Code process inheriting this env won't reap. Setting it
REM from INSIDE Claude Code has no effect per the docs, but from a
REM plain cmd/PowerShell the env propagates cleanly.
set CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1

REM `start "title" /D dir cmd` spawns a new console. /B would hide the
REM window but also re-parent to this shell (defeats the purpose); use
REM a real title instead so the user can find + close the window from
REM Task Manager.
start "agentic-investor: paper-experiment %EXP%" /D "%REPO%" cmd /c ^
  ".venv\Scripts\agentic-investor.exe paper-experiment %EXP% --serve-dashboard --dashboard-port 8000 --paper-loop-args --auto --top-n 8 --regen-mode event --amount 50000 !EXTRA! > out\logs\experiment.out 2>&1"

echo Launched paper-experiment %EXP% in a new console.
echo Log:      %REPO%\out\logs\experiment.out
echo Dashboard: http://localhost:8000/
echo Stop:      Close the "agentic-investor: paper-experiment %EXP%" console window
endlocal
