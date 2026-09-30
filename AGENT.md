# AGENT.md

Guidance for AI coding agents working in this repository. See `README.md` for user-facing docs and controls.

## Project

Neural Circuit: a Pygame race simulation where 50 cars learn to drive through neuroevolution. There is no backpropagation and no pretrained model. Small fully connected `tanh` networks read a fan of distance sensors and output three continuous controls (steering, signed drive, brake).

## Layout

| File | Purpose |
| --- | --- |
| `main.py` | Entry point and simulation. Contains `Track` (road geometry, Catmull-Rom splines, pinched widths), `Car` (physics, sensors), `Race` (a heat: ranking, lap detection), `App` (UI, event loop, road switching). Constants such as `POPULATION`, `ROAD_SPECS`, `SENSOR_RANGE` and `LAPS_PER_ROAD` live at the top. |
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

- Every brain gets `sensors + EXTRA_INPUTS` inputs (the last is the car's speed, `Car.inputs`; `Car.sensors` are only the rays) and has direct input-to-output `Network.skip` weights (`SKIP_CONNECTIONS`). `Race.inputs` and the UI stepper count sensors only; `Race.sizes[0]` includes the speed. Changing the input layout means bumping `SAVE_VERSION`. The output layer always has exactly 3 neurons. Drive is signed (negative means reverse, capped below forward speed). A positive brake output slows the car in either direction.
- Selection uses `Race.fitness(i)`: every brain also drives `PROBE_COUNT` hard probe roads (`Race.probes`, art-less `Track` variants, levels `PROBE_MIN_LEVEL` and up, random orientation, independent of the gauntlet road) in the same heat; fitness = 0.5 × mean + 0.5 × min of distance / `REFERENCE_SPEED × heat_limit`. `Car.rank_key` (laps, then distance) is only used to pick the leader shown on screen. Never rank on the shown road alone: that is how a network memorises a road.
- `Track(level, mirror, reverse)` gives four orientations of each road; `Track.art` is drawn lazily, so only shown roads cost memory. Always look roads up through `Race.track_for(level, mirror, reverse)`.
- There are 11 roads. Roads 1, 2, 10 and 11 keep their original layouts, and `test_race.py` asserts on geometry such as the start point and road widths. Update those tests deliberately when you change `ROAD_SPECS`.
- Gauntlet (`Race`): the champion is always `cars[0]`. It needs `LAPS_PER_ROAD` (5) laps per road, `FINAL_ROAD_LAPS` (10) and a 60 s heat on road 11, then random roads; `tracks_completed` keeps counting. A champion crash or a challenger whose fitness exceeds `(1 + BEAT_MARGIN)` times the champion's resets to road 1 and 0. Always start on road 1.
- `best_network.json` is written only when `best_score_ever` beats the saved score. Tests must never touch the real file: `test_race.py` patches `main.SAVE_PATH` to a temp directory in `setUp`.
- Score: `Car.lap_score` (finished laps) and `Car.score_on` (plus the partial lap), constants `SCORE_*`. `Race.run_score` is the champion's score since it was crowned (`run_banked` plus its live heat) and resets with the gauntlet. `best_score_ever` is the persisted record, tied to `record_brain`; `best_network.json` holds that one model. Higher must always mean a better saved model. After a champion crash the heat continues and `run_score` follows `scoring_car` (the best car still racing, else the best car), so the displayed score never freezes.
- `slot_role()` in `neural.py` must match how `next_generation()` orders cars (2 elites, 2 mutants, children, newcomers): `Race` uses it to say how the winner was made (`Race.last_result`).
- Cars die by leaving the road, stalling (speed < 3 after 8 s), or stagnating (`STAGNATION_DISTANCE` of `best_progress` per `STAGNATION_SECONDS`). Stagnation matters: a champion that circles or oscillates on the road never crashes, is never replaced, and freezes the run. Reversing with no forward progress dies after 4 s by design.
- Sensor noise (`SENSOR_NOISE`) and start-heading jitter are deliberate anti-overfitting measures; set `Car.sensor_noise = 0` when a test needs deterministic sensors.
- Switching roads restarts the heat but keeps the networks. **Apply & Restart** and `R` start a new population.
- The UI palette constants (`BG`, `PANEL`, `TEXT`, `MUTED` and so on) are duplicated in `main.py` and `inspector.py`. Keep them in sync.
- The code is plain Python with type hints and no framework. Match the existing style, and keep changes small and self-contained.
- If you change controls or behaviour, update `README.md` and the tests in the same change.

## Tuning learning speed

Measure changes, do not guess: the useful yardstick is how much of a lap the champion drives on each of the 11 roads in two orientations (about 19 of 22 with the current settings, about 14.5 before hard probes), averaged over several seeds of at least 30 simulated minutes. The gauntlet score alone is too noisy and depends on how often champions are replaced. Things that were tested and did not help on their own: a speed input alone, smaller or larger hidden layers, 9 sensors, lower mutation rates or sizes, less crossover, easy or gauntlet-tied probe roads.

## Workflow

- Run `python -m unittest -v` before finishing any change.
- Don't commit `.venv/` or `__pycache__/` (already in `.gitignore`).
- To check visual or UI changes, launch `python main.py`. Tests don't cover rendering.
