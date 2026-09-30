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
| **Algorithm** | Switch between **Evolution** and **PPO**. Each mode keeps its in-memory training state while you use the other mode. Evolution saves `best_network.json`; PPO saves `ppo_checkpoint.pt`. |
| Road arrows, or left/right arrow keys | Switch among eleven roads. Roads 1, 2, 10, and 11 keep their original layouts. Roads 3–9 have tighter turns that reward braking before a bend, and Roads 6–10 include narrow sections. Switching restarts the current heat on the new road while keeping the networks, and restarts the gauntlet there. |
| Max runtime `−` / `+` | Set a heat's time limit from 5 to 120 simulated seconds. Changes take effect immediately. |
| **Limit On / Limit Off** | Turn the time limit on or off. With it off, the heat continues until every car has left the road, stalled or stopped making progress. |
| Input neurons, hidden layers, layer widths | Change the network design. Click **Apply & Restart** to start a new population with those settings. The output layer always has three neurons. |
| **Pause** / `P` | Pause or resume. |
| **Speed** / `Tab` | Cycle through 1×, 2×, 4×, and 8× simulation speed. Starts at 4×. |
| `V` | Show or hide the leading car's sensor rays. |
| `R` | Restart training with the current settings. |
| **View Network** | Open a live window for the highest scoring evolutionary car or the current PPO policy. The diagram shows every neuron and connection; green and orange lines indicate positive and negative weights. Click a neuron to see its activation, bias, weighted input, and exact connection weights. PPO shows the policy's mean action; exploration can make the car's sampled action differ. |

## How learning works

### PPO

PPO trains one shared actor-critic network with backpropagation. Eight cars collect driving experience in parallel. **The car on screen is worker 1 and is actually contributing to training**; the other seven drive on roads that are simulated without drawing them. The visible worker advances to the next road after five completed runs, cycling through all 11 roads. Choosing a road in the UI starts a fresh five-run count there. Other workers sample roads and orientations from a curriculum that unlocks harder roads as training proceeds.

Each step rewards forward progress, gives a small lap bonus, and penalizes crashes and wasted time. After 256 steps per worker, PPO computes generalized advantage estimates and performs four epochs of clipped policy updates in small batches. Optimization is spread over UI frames; the visible car continues driving while it runs. The network architecture controls also apply to PPO. Switching away pauses its trainer, and returning resumes it. **Apply & Restart** starts a fresh policy for the selected mode.

PPO checkpoints are saved separately to `ppo_checkpoint.pt` (git-ignored), periodically and on exit. A matching checkpoint loads when PPO is first selected after startup. The PPO display shows update count, episodes, best distance on the visible road, and mean episode reward. Rewards are training feedback; they are not the evolutionary run score.

### Evolution

Each evolutionary generation starts with 50 cars. Their sensor distances feed a fully connected network with `tanh` neurons. Cars are ranked by fitness (see below). At the end of a heat, the two best networks pass to the next generation unchanged, two lightly mutated copies of them (small nudges, no crossover) follow, most of the rest are bred and mutated from high-ranking cars, and a few are generated at random. Evolution does not use backpropagation.

### The gauntlet and the counter

The best car (the *champion*) always sits in slot 0 of each generation, unchanged. Training always starts on Road 1. The champion must finish 5 laps on a road (laps carry over between heats) before the next road starts; Roads 1 to 11 run in order, and Road 11 needs 10 laps and gets a 60 second heat. After that, roads are random, and every road the champion completes keeps adding to the **Tracks completed** counter shown on the track and in the panel.

The counter resets to 0 and training restarts on Road 1 whenever the champion crashes (leaves the road, stalls, or goes 4 seconds without gaining 30 px of forward progress) or another car is clearly fitter: more than twice its fitness (see below). The best car of that generation becomes the new champion. Jumping to another road with the road arrows also resets the counter and starts the gauntlet from that road.

### Learning to drive, not to remember a road

A network that only fits one road, or one direction of it, would look good in a single race and then fail elsewhere. Three things prevent that:

- **Probe roads.** In every heat each network also drives two hidden probe roads at the same time, each with its own random start. They are always hard roads (Roads 5 to 11) in a random orientation, whichever road the gauntlet is on. Probe roads are simulated, not drawn; the note at the bottom left lists them. Hard roads have to shape the population from the first generation: training on easy roads first teaches a network to drive flat out everywhere, and it then dies at the first tight bend of the hard roads.
- **Mirrored and reversed roads.** Every road exists in four orientations: original, mirrored (left turns become right turns), reversed (driven the other way round) and both. A network that just leans one way cannot pass all four. After Road 11 the shown road is random in road and orientation too.
- **Fitness across all roads.** Selection uses `0.5 × average + 0.5 × worst` of the distance driven on the shown road and both probe roads, scaled to the same units on every road. A specialist that is brilliant on one road and crashes on the others loses to an all-rounder. The champion's 5-lap gauntlet on 11 different roads is the final exam. A challenger only replaces the champion when its fitness is more than twice the champion's: fitness is noisy and the population keeps improving, and a small margin turned nearly every heat into a takeover that reset the run.

Sensor readings also carry a little noise, and cars start with a small random heading and offset.

### Cars that go nowhere

A car is also removed when it gains less than 30 px of forward progress in 4 seconds. Without this a network can drive in circles or back and forth on the road forever: it never crashes, so a champion doing it is never replaced, and the run and its score freeze on that road.

### The network

Each network is fully connected with `tanh` neurons. Its inputs are the sensor distances plus the car's own speed (signed, divided by top speed); the **Input neurons** control sets the number of sensors, and one speed input is always added. Without speed, a network cannot tell how hard to brake for a bend it is approaching. The inputs also feed the three outputs directly, next to the hidden layers, so simple reflexes such as steering toward the open side or braking when fast with a wall ahead need only a few weights. The **View Network** window draws these direct connections and lists them in the neuron details. Saved networks from before the speed input are ignored.

### Who won the last generation

When a new generation starts, a note at the bottom left says who won the previous one and how that car was made: the unchanged champion, the runner-up (an unchanged copy of the 2nd best), a lightly mutated copy of a top car, a bred child of two top cars, or a brand-new random network. A second line says what that meant for the run: the champion keeps its title, finished a road, crashed, or was overtaken (either of the last two sends training back to Road 1). When the champion crashes the heat carries on until the time limit or the last car stops, and the score switches to the best car still racing, starting from zero, and the note appears when the heat ends.

### Score

A higher score means a better model, so the score belongs to the champion's run rather than to a single heat. Each finished lap earns 100 points plus up to 50 for pace (its lap time against top speed), the lap in progress earns a fraction of 100, and everything is multiplied by `1 + 0.1 × (road − 1)` so later roads are worth more. The **Run score** adds this up across every road the champion completes in a row, so it only grows while the champion keeps driving and keeps going into the random-road phase.

The run score resets to 0 when the champion is dethroned or crashes (the same moment the tracks counter resets). **Best score ever** never resets: it is the highest run score any model has reached. Before the first champion exists, the score is that of the best car.

### Saved network

The model that set the best score ever is saved to `best_network.json` (git-ignored), together with the score and how many tracks that run completed. It is written when a heat ends with a new record and again on quit. On startup the app loads it, adopts its network layout, seeds the population from it and starts on Road 1; if the file is missing or unreadable it starts from scratch. **Restart Training** (`R`) starts from scratch but keeps the record, and the file is only replaced when a new model beats the saved score.

To run the checks:

```sh
python -m unittest -v
```
