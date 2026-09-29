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
| Road arrows, or left/right arrow keys | Switch among eleven roads. Roads 1, 2, 10, and 11 keep their original layouts. Roads 3–9 have tighter turns that reward braking before a bend, and Roads 6–10 include narrow sections. Switching restarts the current heat on the new road while keeping the networks. |
| Max runtime `−` / `+` | Set a heat's time limit from 5 to 120 simulated seconds. Changes take effect immediately. |
| **Limit On / Limit Off** | Turn the time limit on or off. With it off, the heat continues until every car has left the road or stalled. |
| Input neurons, hidden layers, layer widths | Change the network design. Click **Apply & Restart** to start a new population with those settings. The output layer always has three neurons. |
| **Mix On / Mix Off** / `M` | With mixing on (default), every generation runs on a random unlocked road so networks must handle all of them. With it off, training stays on the selected road. |
| **Pause** / `P` | Pause or resume. |
| **Speed** / `Tab` | Cycle through 1×, 2×, 4×, and 8× simulation speed. Starts at 4×. |
| `V` | Show or hide the leading car's sensor rays. |
| `R` | Restart training with the current settings. |
| **View Network** | Open a live window for the highest scoring car in the current generation. The diagram shows every neuron and connection; green and orange lines indicate positive and negative weights. Click a neuron to see its current activation, bias, weighted input, and exact incoming and outgoing connection weights. Scroll the details pane for longer lists. |

## How learning works

Each generation starts with 50 cars. Their sensor distances feed a fully connected network with `tanh` neurons. A car that completes a lap ranks above any car that has not; among finishers, the fastest lap wins. Until a car completes a lap, furthest forward progress determines the ranking. At the end of a heat, the two best networks pass to the next generation unchanged, two lightly mutated copies of them (small nudges, no crossover) follow, most of the rest are bred and mutated from high-ranking cars, and a few are generated at random. This is neuroevolution; the app does not use backpropagation or a pretrained model.

### Generalising across roads

To avoid a network that only knows one road, each generation runs on a random road from the unlocked set (Roads 1 to N), sensor readings carry a little noise, and cars start with a small random heading and offset. Champions are re-tested on a different road every generation, so a driver that only works on one layout drops out.

If at least one car finishes a full lap in each of five consecutive generations, the next road unlocks and is used for the next generation. A generation without a full lap resets the count. Road 11 is the last road. With **Mix Off**, the old behaviour applies: training stays on the selected road and moves to the next one after five lap generations.

### Saved network

When a generation ends with at least one full lap, the two best networks are saved to `best_network.json` (git-ignored), along with how many roads are unlocked. On startup the app loads that file, adopts its network layout, and seeds the population from it; if the file is missing or unreadable it starts from scratch. **Restart Training** (`R`) always starts from scratch, and the file is only replaced again once the new run has unlocked as many roads as the saved one, so a restart does not wipe a good model.

To run the checks:

```sh
python -m unittest -v
```
