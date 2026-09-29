# Neural Circuit

A Pygame race simulation where 50 cars learn to drive by evolving small neural networks. Each car sees the distance to the road edge through a fan of sensors. Its network produces three continuous controls: left/right steering, forward/reverse drive, and a separate brake.

The drive output is signed: positive accelerates forward and negative accelerates in reverse. A positive brake output slows the car in either direction; zero or negative releases the brake. Reverse speed is capped below forward speed.

## Run

Python 3.13 is recommended. Create a virtual environment and install Pygame:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## Controls

| Control | Action |
| --- | --- |
| Road arrows, or left/right arrow keys | Switch among eleven roads. Roads 1, 2, 10, and 11 keep their original layouts. Roads 3–9 have tighter turns that reward braking before a bend, and Roads 6–10 include narrow sections. Switching restarts the current heat on the new road while keeping the networks, and restarts the gauntlet there. |
| Max runtime `−` / `+` | Set a heat's time limit from 5 to 120 simulated seconds. Changes take effect immediately. |
| **Limit On / Limit Off** | Turn the time limit on or off. With it off, the heat continues until every car has left the road or stalled. |
| Input neurons, hidden layers, layer widths | Change the network design. Click **Apply & Restart** to start a new population with those settings. The output layer always has three neurons. |
| **Pause** / `P` | Pause or resume. |
| **Speed** / `Tab` | Cycle through 1×, 2×, 4×, and 8× simulation speed. Starts at 4×. |
| `V` | Show or hide the leading car's sensor rays. |
| `R` | Restart training with the current settings. |
| **View Network** | Open a live window for the highest scoring car in the current generation. The diagram shows every neuron and connection; green and orange lines indicate positive and negative weights. Click a neuron to see its current activation, bias, weighted input, and exact incoming and outgoing connection weights. Scroll the details pane for longer lists. |

## How learning works

Each generation starts with 50 cars. Their sensor distances feed a fully connected network with `tanh` neurons. A car that completes a lap ranks above any car that has not; among finishers, the fastest lap wins. Until a car completes a lap, furthest forward progress determines the ranking. At the end of a heat, the two best networks pass to the next generation unchanged, two lightly mutated copies of them (small nudges, no crossover) follow, most of the rest are bred and mutated from high-ranking cars, and a few are generated at random. This is neuroevolution; the app does not use backpropagation or a pretrained model.

### The gauntlet and the counter

The best car (the *champion*) always sits in slot 0 of each generation, unchanged. Training always starts on Road 1. The champion must finish 5 laps on a road (laps carry over between heats) before the next road starts; Roads 1 to 11 run in order, and Road 11 needs 10 laps and gets a 60 second heat. After that, roads are random, and every road the champion completes keeps adding to the **Tracks completed** counter shown on the track and in the panel.

The counter resets to 0 and training restarts on Road 1 whenever the champion crashes (leaves the road or stalls) or another car clearly beats it: it finishes a lap when the champion has not, or its fastest lap is at least 3% faster. The best car of that generation becomes the new champion. Jumping to another road with the road arrows also resets the counter and starts the gauntlet from that road.

To avoid overfitting one road, sensor readings carry a little noise and cars start with a small random heading and offset.

### Who won the last generation

When a new generation starts, a note at the bottom left says who won the previous one and how that car was made: the unchanged champion, the runner-up (an unchanged copy of the 2nd best), a lightly mutated copy of a top car, a bred child of two top cars, or a brand-new random network. A second line says what that meant for the run: the champion keeps its title, finished a road, crashed, or was overtaken (either of the last two sends training back to Road 1). A champion crash ends the heat at once, since the run is already lost and its score cannot change.

### Score

A higher score means a better model, so the score belongs to the champion's run rather than to a single heat. Each finished lap earns 100 points plus up to 50 for pace (its lap time against top speed), the lap in progress earns a fraction of 100, and everything is multiplied by `1 + 0.1 × (road − 1)` so later roads are worth more. The **Run score** adds this up across every road the champion completes in a row, so it only grows while the champion keeps driving and keeps going into the random-road phase.

The run score resets to 0 when the champion is dethroned or crashes (the same moment the tracks counter resets). **Best score ever** never resets: it is the highest run score any model has reached, Before the first champion exists, the score is that of the best car.

### Saved network

The model that set the best score ever is saved to `best_network.json` (git-ignored), together with the score and how many tracks that run completed. It is written when a heat ends with a new record and again on quit. On startup the app loads it, adopts its network layout, seeds the population from it and starts on Road 1; if the file is missing or unreadable it starts from scratch. **Restart Training** (`R`) starts from scratch but keeps the record, and the file is only replaced when a new model beats the saved score.

To run the checks:

```sh
python -m unittest -v
```
