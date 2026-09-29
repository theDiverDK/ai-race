# AGENT.md

Guidance for AI coding agents working in this repository. See `README.md` for user-facing docs and controls.

## Project

Neural Circuit: a Pygame race simulation where 50 cars learn to drive through neuroevolution. There is no backpropagation and no pretrained model. Small fully connected `tanh` networks read a fan of distance sensors and output three continuous controls (steering, signed drive, brake).

## Layout

| File | Purpose |
| --- | --- |
| `main.py` | Entry point and simulation. Contains `Track` (road geometry, Catmull-Rom splines, pinched widths), `Car` (physics, sensors), `Race` (a heat: ranking, lap detection), `App` (UI, event loop, road switching). Constants such as `POPULATION`, `ROAD_SPECS`, `SENSOR_RANGE` and `LAP_GENERATIONS_TO_ADVANCE` live at the top. |
| `neural.py` | `Network` class, `next_generation()` (2 unchanged champions, 2 `minor_mutation` copies, bred children, random newcomers), and `save_networks()` / `load_networks()` for `best_network.json`. |
| `inspector.py` | `NetworkInspector`, the live network-visualisation window, and `activations()`. |
| `test_race.py`, `test_neural.py` | `unittest` suites. |

## Commands

```sh
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt   # pygame>=2.6,<3
python main.py                              # run the app
python -m unittest -v                       # run all tests
```

Tests set `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`, so they run headless. Keep that setup at the top of any new test file that imports `main`.

## Conventions and invariants

- The output layer always has exactly 3 neurons. Drive is signed (negative means reverse, capped below forward speed). A positive brake output slows the car in either direction.
- Ranking: a car that completes a lap beats any car that has not, and finishers are ordered by fastest lap. Otherwise furthest forward progress decides.
- There are 11 roads. Roads 1, 2, 10 and 11 keep their original layouts, and `test_race.py` asserts on geometry such as the start point and road widths. Update those tests deliberately when you change `ROAD_SPECS`.
- Gauntlet (`Race`): the champion is always `cars[0]`. It needs `LAPS_PER_ROAD` (5) laps per road, `FINAL_ROAD_LAPS` (10) and a 60 s heat on road 11, then random roads; `tracks_completed` keeps counting. A champion crash or a challenger that beats it by `BEAT_MARGIN` resets to road 1 and 0. Always start on road 1.
- `best_network.json` is written only when `best_score_ever` beats the saved score. Tests must never touch the real file: `test_race.py` patches `main.SAVE_PATH` to a temp directory in `setUp`.
- Score: `Car.lap_score` (finished laps) and `Car.score_on` (plus the partial lap), constants `SCORE_*`. `Race.run_score` is the champion's score since it was crowned (`run_banked` plus its live heat) and resets with the gauntlet. `best_score_ever` is the persisted record, tied to `record_brain`; `best_network.json` holds that one model. Higher must always mean a better saved model.
- Sensor noise (`SENSOR_NOISE`) and start-heading jitter are deliberate anti-overfitting measures; set `Car.sensor_noise = 0` when a test needs deterministic sensors.
- Switching roads restarts the heat but keeps the networks. **Apply & Restart** and `R` start a new population.
- The UI palette constants (`BG`, `PANEL`, `TEXT`, `MUTED` and so on) are duplicated in `main.py` and `inspector.py`. Keep them in sync.
- The code is plain Python with type hints and no framework. Match the existing style, and keep changes small and self-contained.
- If you change controls or behaviour, update `README.md` and the tests in the same change.

## Workflow

- Run `python -m unittest -v` before finishing any change.
- Don't commit `.venv/` or `__pycache__/` (already in `.gitignore`).
- To check visual or UI changes, launch `python main.py`. Tests don't cover rendering.
