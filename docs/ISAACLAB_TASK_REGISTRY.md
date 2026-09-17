# Installed IsaacLab task registry

The HPC IsaacLab 2.3.0 environment exposed 152 `Isaac-*` Gym registrations
when queried on an allocated GPU node on 2026-09-17. Registration does not by
itself imply compatibility with M4PO's vector-only sequential adapter: camera,
discrete-action, multi-agent, deployment, and demonstration-oriented tasks
need additional integration work.

## Selected Cartpole pair

The fast multi-task configuration uses two implementations of the same
Cartpole embodiment:

| Registry ID | Implementation | Policy observation | Action | Native horizon |
|---|---|---:|---:|---:|
| `Isaac-Cartpole-Direct-v0` | `DirectRLEnv` | 4 | 1 | 300 |
| `Isaac-Cartpole-v0` | `ManagerBasedRLEnv` | 4 | 1 | 300 |

Both use continuous vector spaces, simulation `dt=1/120`, and control
decimation 2. The Direct observation is ordered as pole position/velocity then
cart position/velocity; the manager-based registration concatenates its joint
positions then joint velocities. M4PO therefore keeps their task contexts
distinct even though the action and embodiment are shared.

For comparable success rates, the adapter adds the Direct task's
`|pole angle| > pi/2` failure condition to the manager-based configuration;
both tasks then count success only when they reach the configured horizon
without a cart- or pole-limit termination. It also applies each API's verified
episode-length offset so both emit their timeout transition on exactly the
requested outer step. Using both registrations exercises Direct and
manager-based IsaacLab integration while keeping the physical task and action
interface fixed.

`m4po/configs/isaaclab_cartpole.yaml` shortens the horizon to 128 outer steps
and collects 64 complete 32-environment rollouts, balanced to 32 rollouts per
registration. `m4po/configs/isaaclab_cartpole_smoke.yaml` uses four tiny
rollouts, covering each registration twice, followed by frozen evaluation of
both tasks through `scripts/smoke_test_isaaclab_cartpole.sh`.

## Family counts

| Family | Count | Family | Count |
|---|---:|---|---:|
| Ant | 2 | AutoMate | 2 |
| Cart Double Pendulum | 1 | Cartpole | 32 |
| Deploy | 3 | Dexsuite | 4 |
| Factory | 3 | Forge | 3 |
| Franka Cabinet | 1 | Humanoid | 5 |
| Lift | 5 | Navigation | 2 |
| Open Drawer | 4 | Place | 2 |
| Quadcopter | 1 | Reach | 8 |
| Repose | 10 | Shadow Hand | 1 |
| Stack | 17 | Tracking | 2 |
| Velocity | 44 | **Total** | **152** |

## Full registry

### Ant (2)

```text
Isaac-Ant-Direct-v0
Isaac-Ant-v0
```

### AutoMate (2)

```text
Isaac-AutoMate-Assembly-Direct-v0
Isaac-AutoMate-Disassembly-Direct-v0
```

### Cart Double Pendulum (1)

```text
Isaac-Cart-Double-Pendulum-Direct-v0
```

### Cartpole (32)

```text
Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0
Isaac-Cartpole-Camera-Showcase-Box-Discrete-Direct-v0
Isaac-Cartpole-Camera-Showcase-Box-MultiDiscrete-Direct-v0
Isaac-Cartpole-Camera-Showcase-Dict-Box-Direct-v0
Isaac-Cartpole-Camera-Showcase-Dict-Discrete-Direct-v0
Isaac-Cartpole-Camera-Showcase-Dict-MultiDiscrete-Direct-v0
Isaac-Cartpole-Camera-Showcase-Tuple-Box-Direct-v0
Isaac-Cartpole-Camera-Showcase-Tuple-Discrete-Direct-v0
Isaac-Cartpole-Camera-Showcase-Tuple-MultiDiscrete-Direct-v0
Isaac-Cartpole-Depth-Camera-Direct-v0
Isaac-Cartpole-Depth-v0
Isaac-Cartpole-Direct-v0
Isaac-Cartpole-RGB-Camera-Direct-v0
Isaac-Cartpole-RGB-ResNet18-v0
Isaac-Cartpole-RGB-TheiaTiny-v0
Isaac-Cartpole-RGB-v0
Isaac-Cartpole-Showcase-Box-Box-Direct-v0
Isaac-Cartpole-Showcase-Box-Discrete-Direct-v0
Isaac-Cartpole-Showcase-Box-MultiDiscrete-Direct-v0
Isaac-Cartpole-Showcase-Dict-Box-Direct-v0
Isaac-Cartpole-Showcase-Dict-Discrete-Direct-v0
Isaac-Cartpole-Showcase-Dict-MultiDiscrete-Direct-v0
Isaac-Cartpole-Showcase-Discrete-Box-Direct-v0
Isaac-Cartpole-Showcase-Discrete-Discrete-Direct-v0
Isaac-Cartpole-Showcase-Discrete-MultiDiscrete-Direct-v0
Isaac-Cartpole-Showcase-MultiDiscrete-Box-Direct-v0
Isaac-Cartpole-Showcase-MultiDiscrete-Discrete-Direct-v0
Isaac-Cartpole-Showcase-MultiDiscrete-MultiDiscrete-Direct-v0
Isaac-Cartpole-Showcase-Tuple-Box-Direct-v0
Isaac-Cartpole-Showcase-Tuple-Discrete-Direct-v0
Isaac-Cartpole-Showcase-Tuple-MultiDiscrete-Direct-v0
Isaac-Cartpole-v0
```

### Deploy (3)

```text
Isaac-Deploy-Reach-UR10e-Play-v0
Isaac-Deploy-Reach-UR10e-ROS-Inference-v0
Isaac-Deploy-Reach-UR10e-v0
```

### Dexsuite (4)

```text
Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0
Isaac-Dexsuite-Kuka-Allegro-Lift-v0
Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0
Isaac-Dexsuite-Kuka-Allegro-Reorient-v0
```

### Factory (3)

```text
Isaac-Factory-GearMesh-Direct-v0
Isaac-Factory-NutThread-Direct-v0
Isaac-Factory-PegInsert-Direct-v0
```

### Forge (3)

```text
Isaac-Forge-GearMesh-Direct-v0
Isaac-Forge-NutThread-Direct-v0
Isaac-Forge-PegInsert-Direct-v0
```

### Franka Cabinet (1)

```text
Isaac-Franka-Cabinet-Direct-v0
```

### Humanoid (5)

```text
Isaac-Humanoid-AMP-Dance-Direct-v0
Isaac-Humanoid-AMP-Run-Direct-v0
Isaac-Humanoid-AMP-Walk-Direct-v0
Isaac-Humanoid-Direct-v0
Isaac-Humanoid-v0
```

### Lift (5)

```text
Isaac-Lift-Cube-Franka-IK-Abs-v0
Isaac-Lift-Cube-Franka-IK-Rel-v0
Isaac-Lift-Cube-Franka-Play-v0
Isaac-Lift-Cube-Franka-v0
Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0
```

### Navigation (2)

```text
Isaac-Navigation-Flat-Anymal-C-Play-v0
Isaac-Navigation-Flat-Anymal-C-v0
```

### Open Drawer (4)

```text
Isaac-Open-Drawer-Franka-IK-Abs-v0
Isaac-Open-Drawer-Franka-IK-Rel-v0
Isaac-Open-Drawer-Franka-Play-v0
Isaac-Open-Drawer-Franka-v0
```

### Place (2)

```text
Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0
Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
```

### Quadcopter (1)

```text
Isaac-Quadcopter-Direct-v0
```

### Reach (8)

```text
Isaac-Reach-Franka-IK-Abs-v0
Isaac-Reach-Franka-IK-Rel-v0
Isaac-Reach-Franka-OSC-Play-v0
Isaac-Reach-Franka-OSC-v0
Isaac-Reach-Franka-Play-v0
Isaac-Reach-Franka-v0
Isaac-Reach-UR10-Play-v0
Isaac-Reach-UR10-v0
```

### Repose (10)

```text
Isaac-Repose-Cube-Allegro-Direct-v0
Isaac-Repose-Cube-Allegro-NoVelObs-Play-v0
Isaac-Repose-Cube-Allegro-NoVelObs-v0
Isaac-Repose-Cube-Allegro-Play-v0
Isaac-Repose-Cube-Allegro-v0
Isaac-Repose-Cube-Shadow-Direct-v0
Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0
Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0
Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0
Isaac-Repose-Cube-Shadow-Vision-Direct-v0
```

### Shadow Hand (1)

```text
Isaac-Shadow-Hand-Over-Direct-v0
```

### Stack (17)

```text
Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0
Isaac-Stack-Cube-Franka-IK-Abs-v0
Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0
Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0
Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0
Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0
Isaac-Stack-Cube-Franka-IK-Rel-v0
Isaac-Stack-Cube-Franka-v0
Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0
Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-Joint-Position-Play-v0
Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-RmpFlow-Play-v0
Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0
Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-v0
Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0
Isaac-Stack-Cube-Instance-Randomize-Franka-v0
Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0
Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0
```

### Tracking (2)

```text
Isaac-Tracking-LocoManip-Digit-Play-v0
Isaac-Tracking-LocoManip-Digit-v0
```

### Velocity (44)

```text
Isaac-Velocity-Flat-Anymal-B-Play-v0
Isaac-Velocity-Flat-Anymal-B-v0
Isaac-Velocity-Flat-Anymal-C-Direct-v0
Isaac-Velocity-Flat-Anymal-C-Play-v0
Isaac-Velocity-Flat-Anymal-C-v0
Isaac-Velocity-Flat-Anymal-D-Play-v0
Isaac-Velocity-Flat-Anymal-D-v0
Isaac-Velocity-Flat-Cassie-Play-v0
Isaac-Velocity-Flat-Cassie-v0
Isaac-Velocity-Flat-Digit-Play-v0
Isaac-Velocity-Flat-Digit-v0
Isaac-Velocity-Flat-G1-Play-v0
Isaac-Velocity-Flat-G1-v0
Isaac-Velocity-Flat-H1-Play-v0
Isaac-Velocity-Flat-H1-v0
Isaac-Velocity-Flat-Spot-Play-v0
Isaac-Velocity-Flat-Spot-v0
Isaac-Velocity-Flat-Unitree-A1-Play-v0
Isaac-Velocity-Flat-Unitree-A1-v0
Isaac-Velocity-Flat-Unitree-Go1-Play-v0
Isaac-Velocity-Flat-Unitree-Go1-v0
Isaac-Velocity-Flat-Unitree-Go2-Play-v0
Isaac-Velocity-Flat-Unitree-Go2-v0
Isaac-Velocity-Rough-Anymal-B-Play-v0
Isaac-Velocity-Rough-Anymal-B-v0
Isaac-Velocity-Rough-Anymal-C-Direct-v0
Isaac-Velocity-Rough-Anymal-C-Play-v0
Isaac-Velocity-Rough-Anymal-C-v0
Isaac-Velocity-Rough-Anymal-D-Play-v0
Isaac-Velocity-Rough-Anymal-D-v0
Isaac-Velocity-Rough-Cassie-Play-v0
Isaac-Velocity-Rough-Cassie-v0
Isaac-Velocity-Rough-Digit-Play-v0
Isaac-Velocity-Rough-Digit-v0
Isaac-Velocity-Rough-G1-Play-v0
Isaac-Velocity-Rough-G1-v0
Isaac-Velocity-Rough-H1-Play-v0
Isaac-Velocity-Rough-H1-v0
Isaac-Velocity-Rough-Unitree-A1-Play-v0
Isaac-Velocity-Rough-Unitree-A1-v0
Isaac-Velocity-Rough-Unitree-Go1-Play-v0
Isaac-Velocity-Rough-Unitree-Go1-v0
Isaac-Velocity-Rough-Unitree-Go2-Play-v0
Isaac-Velocity-Rough-Unitree-Go2-v0
```
