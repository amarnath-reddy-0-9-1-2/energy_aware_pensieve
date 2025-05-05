# Energy-Aware ABR using Reinforcement Learning (Pensieve Extension)

This project is an energy-aware extension of the Pensieve ABR (Adaptive Bitrate) streaming framework. It modifies the original reward function to include energy consumption as a penalty, allowing the model to learn streaming strategies that balance Quality of Experience (QoE) and energy efficiency.

##  Prerequisites

- **Operating System**: Ubuntu 16.04 (recommended)
- **Python Version**: Python 2.7 (with TensorFlow and Keras compatible versions)
- **Dependencies**:
  - `matplotlib`
  - `tensorflow`
  - `keras`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `tcpdump`
  - EnergyBox setup (external tool)

##  How to Use

### 1. Train the Energy-Aware RL Model

Modify the energy penalty by setting the `energy_impact` value in `multi_agent.py`.

```bash
python multi_agent.py
```

This trains the A3C model with the specified energy impact coefficient and saves the learned policy.

### 2. Evaluate the Trained Model

Run the trained RL model on different network traces to generate evaluation logs:

```bash
python test.py
```

Ensure the trace files (e.g., FCC, Belgium, 3G/4G) are available in the `traces/` directory.

### 3. Measure Download Energy

Start the RL streaming server:

```bash
python rl_server.py
```

Then, use `tcpdump` to capture traffic while the video is streamed:

```bash
sudo tcpdump -i any -w pcap/trace_output.pcap
```

After capture, use **EnergyBox** to analyze the `.pcap` file and calculate download energy based on the network behavior and device configuration.

### 4. Estimate Playback Energy

Run the script to calculate playback energy per chunk using a trained power model:

```bash
python calculate_eplay.py
```

Make sure the input dataset includes video-level metadata such as bitrate, resolution, motion, quality, and pixel count.

### 5. Train a New Playback Power Model (Optional)

If new energy data is available from another device or experiment:

```bash
python power_model/train.py
```

This will train a regression model that predicts playback power consumption using video features and measured power traces.

##  Output

- Energy and QoE logs
- Bitrate transition behavior
- Total energy breakdown: Download + Playback
- Model behavior across energy impact values (e.g., 0.05, 0.10, 0.20)

##  Notes

- Packet captures are essential for accurate download energy estimation.
- Playback energy model must be trained per device to reflect actual power characteristics.
- The energy-aware model shows smoother bitrate transitions and better energy-QoE trade-offs at higher energy penalties.
