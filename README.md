# AETHERGRID

## Autonomous Edge Distributed AI Compute Fabric

<p align="center">
A Distributed Intelligence System Built from Laptop + Raspberry Pi
</p>

Author
**C. Kumaran**

---

# Overview

AETHERGRID is a lightweight distributed computing framework that transforms small devices into a collaborative compute cluster.

The system allows a laptop and Raspberry Pi to work together to execute tasks such as:

* distributed AI inference
* Monte Carlo simulations
* particle simulations
* parallel data processing

Unlike large distributed frameworks, AETHERGRID is designed to run on **low-power hardware** such as a Raspberry Pi 3.

The laptop acts as the **control node**, while Raspberry Pi devices operate as **worker nodes**.

---

# Hardware Setup

Cluster Configuration

Control Node
Laptop
Intel i5-12450H
16GB RAM
Linux (EndeavourOS)

Worker Node
Raspberry Pi 3 Model B
Quad-core ARM Cortex-A53
1GB RAM

Both devices must be connected to the **same WiFi or LAN network**.

---

# System Architecture

```
                AETHERGRID CLUSTER

          ┌──────────────────────────┐
          │        LAPTOP            │
          │      CONTROL NODE        │
          │                          │
          │ Task Scheduler           │
          │ Dataset Coordinator      │
          │ Result Aggregator        │
          └──────────────┬───────────┘
                         │
                    Local Network
                         │
                 ┌───────┴────────┐
              

          ┌───────────────┐
          │  RASPBERRY PI │
          │  WORKER NODE  │
          │               │
          │ Task Executor │
          │ Resource Mon. │
          └───────────────┘
```

---

# Cluster Operation

```
Node Startup
     │
     ▼
Auto Discovery
     │
     ▼
Node Registration
     │
     ▼
Task Assignment
     │
     ▼
Parallel Execution
     │
     ▼
Return Results
```

The worker node continuously listens for tasks from the laptop.

---

# Mathematical Model for Distributed Workload

T = \frac{W}{N \cdot C}

Where

T = execution time
W = workload size
N = number of nodes
C = compute capacity of each node

With two nodes (Laptop + Pi), execution time decreases depending on the workload.

---

# Auto Node Discovery

Workers automatically find the control node using UDP broadcast.

Worker startup flow

```
Worker Boot
     │
     ▼
Broadcast Discovery Packet
     │
     ▼
Laptop Responds
     │
     ▼
Worker Registers
```

This means you **do not need to manually configure IP addresses**.

---

# Distributed AI Inference

Example: MNIST image classification

Dataset size: 60000 images

```
Laptop → images 1–40000
Pi     → images 40001–60000
```

Both devices perform inference simultaneously.

Results are merged by the laptop.

---

# Monte Carlo Simulation

Example

```
Simulation Trials = 1,000,000

Laptop → 700,000 trials
Pi     → 300,000 trials
```

Results are combined to compute probability estimates.

---

# Cluster Visualization

The cluster includes a simple terminal-based visualization.

Example

```
CLUSTER STATUS

Node            CPU      Tasks
--------------------------------
Laptop          42%      3
RaspberryPi     31%      1
```

This runs directly on the laptop.

---

# Performance Example

Single Laptop

100 seconds

Laptop + Pi

65 seconds

Speed improvement ≈ **1.5×**

This varies depending on workload.

---

# Repository Structure

```
aethergrid

control_plane
  scheduler.py
  node_registry.py
  task_dispatcher.py

worker_runtime
  worker_node.py
  task_executor.py

networking
  discovery.py
  protocol.py

distributed_ai
  mnist_inference.py

simulation
  monte_carlo.py
  particle_simulation.py

monitoring
  cluster_status.py

experiments
  run_experiment.py
```

---

# Installation

Clone repository

```
git clone https://github.com/ckumaran/aethergrid.git
cd aethergrid
```

Install dependencies

```
pip install numpy flask psutil scikit-learn
```

These libraries run fine on Raspberry Pi.

---

# Running the Cluster

Start the control node on the laptop

```
python control_plane/scheduler.py
```

Start the worker node on Raspberry Pi

```
python worker_runtime/worker_node.py
```

The worker will automatically connect to the cluster.

---

# Running Experiments

Example distributed experiment

```
python experiments/run_experiment.py
```

This launches a distributed Monte Carlo simulation.

---

# Running Another Project on Raspberry Pi

Since Raspberry Pi 3 has limited RAM, the worker should run with limited CPU priority.

Run worker node using

```
nice -n 10 python worker_runtime/worker_node.py
```

Or restrict CPU cores

```
taskset -c 0,1 python worker_runtime/worker_node.py
```

This allows another project to run on the Pi simultaneously.

---

# System Limitations

Raspberry Pi 3 has only 1GB RAM.

Recommended workloads

AI inference
Monte Carlo simulations
parallel data processing

Avoid

large neural network training
large datasets exceeding memory

---

# Future Improvements

Possible extensions

distributed reinforcement learning
web-based cluster dashboard
multiple Raspberry Pi nodes
internet-scale distributed clusters

---

# Author

C. Kumaran

Distributed Systems Experiment
Edge AI Infrastructure

---

# License

MIT License
