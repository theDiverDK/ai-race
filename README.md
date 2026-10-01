# Neural Circuit

A Pygame race simulation with two selectable training algorithms: neuroevolution and Proximal Policy Optimization (PPO). Each car sees the distance to the road edge through a fan of sensors and feels its own speed. Its network produces three continuous controls: left/right steering, forward/reverse drive, and a separate brake.

The drive output is signed: positive accelerates forward and negative accelerates in reverse. A positive brake output slows the car in either direction; zero or negative releases the brake. Reverse speed is capped below forward speed.

## Run

Python 3.13 is recommended. Create a virtual environment and install Pygame, NumPy, and PyTorch:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## Controls

| Control | Action |
| --- | --- |
| **Algorithm dropdown** | Choose **Evolution** or **PPO**. The selected mode is saved in `app_settings.json` and restored on startup. Both modes use the same control layout, and each keeps its in-memory training state while you use the other. Evolution saves `best_network.json`; PPO saves `ppo_checkpoint.pt`. |
| Road arrows, or left/right arrow keys | Switch among eleven roads. Roads 1, 2, 10, and 11 keep their original layouts. Roads 3–9 have tighter turns that reward braking before a bend, and Roads 6–10 include narrow sections. Switching restarts the current heat on the new road while keeping the networks, and restarts the gauntlet there. |
| Max runtime `−` / `+` | Set the optional heat time limit from 5 to 120 simulated seconds. Changes take effect immediately when the limit is on. |
| **Limit On / Limit Off** | Turn the time limit on or off. Evolution starts with the limit off: a generation continues until a living car finishes five laps or all cars stop. Turn it on to cap a generation at the selected runtime. PPO starts with the limit on; disabling it pauses automatic road progression. |
| Input neurons, hidden layers, layer widths | Change the network design. Click **Apply & Restart** to start a new population with those settings. The output layer always has three neurons. |
| **Pause** / `P` | Pause or resume. |
| **Speed** / `Tab` | Cycle through 1×, 2×, 4×, and 8× simulation speed. Starts at 4×. |
| **Debug** | Save `debug_snapshot.json` beside `best_network.json`. It records the selected road, settings, all cars, and the last 20 Evolution generation changes with their exact end reasons and car states. The file is ignored by Git and can be shared when diagnosing a surprising reset. |
| `V` | Show or hide the leading car's sensor rays. |
| `R` | Restart training with the current settings. |
| **View Network** | Open a live window for the highest scoring evolutionary car or the current PPO policy. The diagram shows every neuron and connection; green and orange lines indicate positive and negative weights. Click a neuron to see its activation, bias, weighted input, and exact connection weights. PPO shows the policy's mean action; exploration can make the car's sampled action differ. |

## How learning works

### PPO

PPO trains one shared actor-critic network with backpropagation. Eight cars collect driving experience in parallel, all on the displayed road. Each car samples its own controls, and all eight are visible and contribute to training. A round counts as one clean run when at least one car reaches the enabled time limit while still driving. Several cars finishing together still count as one run. The whole group restarts after a clean run; after five, it moves to the next road, cycling through all 11 roads. Crashes alone do not advance the road counter. Choosing a road in the UI resets the count and moves the whole group there. With the time limit disabled, runs continue until a car crashes, so the road stays selected until the limit is enabled or you choose another road.

Each step rewards forward progress, gives a small lap bonus, and penalizes crashes and wasted time. After 256 steps per worker, PPO computes generalized advantage estimates and performs four epochs of clipped policy updates in small batches. Optimization is spread over UI frames; all eight cars continue driving while it runs. The network architecture controls also apply to PPO. Switching away pauses its trainer, and returning resumes it. **Apply & Restart** starts a fresh policy for the selected mode.

PPO checkpoints are saved separately to `ppo_checkpoint.pt` (git-ignored), periodically and on exit. A matching checkpoint loads when PPO is first selected after startup. The PPO display shows update count, episodes, best distance on the visible road, and mean episode reward. Rewards are training feedback; they are not the evolutionary run score.

### Evolution

Each evolutionary generation starts with 50 cars. Their sensor distances feed a fully connected network with `tanh` neurons. Cars are ranked by fitness (see below). At the end of a heat, the two best networks pass to the next generation unchanged, two lightly mutated copies of them (small nudges, no crossover) follow, most of the rest are bred and mutated from high-ranking cars, and a few are generated at random. Evolution does not use backpropagation.

### The gauntlet and the counter

The best car (the *champion*) always sits in slot 0 of each generation, unchanged. Training always starts on Road 1. A generation ends when a living car finishes five laps, all cars stop, or the optional time limit expires. The champion must finish 5 laps on a road (laps carry over between generations when the time limit is enabled) before the next road starts; Roads 1 to 11 run in order. Road 11 also needs five laps. Its optional time limit is at least 60 seconds. After that, roads are random, and every road the champion completes keeps adding to the **Tracks completed** counter shown on the track and in the panel.

The counter resets to 0 and training restarts on Road 1 whenever the champion crashes (leaves the road, stalls, or goes 4 seconds without gaining 30 px of forward progress) or another car is clearly fitter: more than twice its fitness (see below). The best car of that generation becomes the new champion. Jumping to another road with the road arrows also resets the counter and starts the gauntlet from that road.

### Training on the displayed road

All 50 evolutionary cars drive only the road you see. Selection compares the distance each car reaches on that road, scaled by the heat limit. A challenger replaces the champion only when its fitness is more than twice the champion's. This margin prevents frequent takeovers from resetting the gauntlet.

The gauntlet moves the displayed road forward as the champion completes it. After Road 11, the displayed road and its orientation vary randomly. Each road can be mirrored and reversed, but cars drive those variants only when one is shown.

Sensor readings also carry a little noise, and cars start with a small random heading and offset. Training on one road at a time can produce drivers that specialize in the current road.

### Cars that go nowhere

A car is also removed when it gains less than 30 px of forward progress in 4 seconds. Without this a network can drive in circles or back and forth on the road forever: it never crashes, so a champion doing it is never replaced, and the run and its score freeze on that road.

### The network

Each network is fully connected with `tanh` neurons. Its inputs are the sensor distances plus the car's own speed (signed, divided by top speed); the **Input neurons** control sets the number of sensors, and one speed input is always added. Without speed, a network cannot tell how hard to brake for a bend it is approaching. The inputs also feed the three outputs directly, next to the hidden layers, so simple reflexes such as steering toward the open side or braking when fast with a wall ahead need only a few weights. The **View Network** window draws these direct connections and lists them in the neuron details. Saved networks from before the speed input are ignored.

### Who won the last generation

When a new generation starts, a note at the bottom left says who won the previous one and how that car was made: the unchanged champion, the runner-up (an unchanged copy of the 2nd best), a lightly mutated copy of a top car, a bred child of two top cars, or a brand-new random network. Another line says what that meant for the run: the champion keeps its title, finished a road, crashed, or was overtaken (either of the last two sends training back to Road 1). A third line says why the generation ended: five laps, time limit, or all cars stopped. When the champion crashes the heat carries on until another car finishes five laps, the enabled time limit expires, or the last car stops. The score switches to the best car still racing, starting from zero, and the note appears when the heat ends.

### Score

A higher score means a better model, so the score belongs to the champion's run rather than to a single heat. Each finished lap earns 100 points plus up to 50 for pace (its lap time against top speed), the lap in progress earns a fraction of 100, and everything is multiplied by `1 + 0.1 × (road − 1)` so later roads are worth more. The **Run score** adds this up across every road the champion completes in a row, so it only grows while the champion keeps driving and keeps going into the random-road phase.

The run score resets to 0 when the champion is dethroned or crashes (the same moment the tracks counter resets). **Best score ever** never resets: it is the highest run score any model has reached. Before the first champion exists, the score is that of the best car.

### Saved network

The model that set the best score ever is saved to `best_network.json` (git-ignored), together with the score and how many tracks that run completed. It is written when a heat ends with a new record and again on quit. On startup the app loads it, adopts its network layout, seeds the population from it and starts on Road 1; if the file is missing or unreadable it starts from scratch. **Restart Training** (`R`) starts from scratch but keeps the record, and the file is only replaced when a new model beats the saved score.

To run the checks:

```sh
python -m unittest -v
```
