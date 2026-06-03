v11 worker count fix

Notebook Cell 5 fix:
- WORKER_COUNT input always wins.
- Preset WORKER_COUNT no longer silently overrides the user value.
- Added suggested presets GPU_STANDARD_MULTI_WORKER_6 and GPU_STANDARD_MULTI_WORKER_8.

Package app.py/pure_runner.py are unchanged from v10 one-line progress.
