# CPSC 4420 HW2: Q-Learning on CliffWalking

Tabular Q-learning on Gymnasium's `CliffWalking-v1`, with experiments on learning rate, discount factor, and ε decay. Has greedy evaluation and video of trained agent. The full analysis is in the PDF report.

## Setup

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux:          source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
python experiments.py     # trains agents, saves plots and Q-tables to results/, prints evaluation
python record_video.py    # records results/agent.mp4 from the saved Q-table
```