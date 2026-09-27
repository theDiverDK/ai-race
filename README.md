# Neural Circuit

A Pygame race simulation where 50 cars learn to drive by evolving small neural networks. Each car sees the distance to the road edge through a fan of sensors. Its network produces two continuous controls: left/right steering and acceleration/braking.

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
| Road arrows, or left/right arrow keys | Switch among ten roads. Roads 6–10 have sharper and more frequent bends. Switching restarts the current heat on the new road while keeping the networks. |
| Max runtime `−` / `+` | Set a heat's time limit from 5 to 120 simulated seconds. Changes take effect immediately. |
| **Limit On / Limit Off** | Turn the time limit on or off. With it off, the heat continues until every car has left the road or stalled. |
| Input neurons, hidden layers, layer widths | Change the network design. Click **Apply & Restart** to start a new population with those settings. The output layer always has two neurons. |
| **Pause** / `P` | Pause or resume. |
| **Speed** / `Tab` | Cycle through 1×, 2×, and 4× simulation speed. |
| `V` | Show or hide the leading car's sensor rays. |
| `R` | Restart training with the current settings. |

## How learning works

Each generation starts with 50 cars. Their sensor distances feed a fully connected network with `tanh` neurons. Cars earn a score based mainly on their furthest forward progress along the road. At the end of a heat, the two best networks are kept, most new networks are bred and mutated from high-scoring cars, and a few are generated at random. This is neuroevolution; the app does not use backpropagation or a pretrained model.

To run the checks:

```sh
python -m unittest -v
```
