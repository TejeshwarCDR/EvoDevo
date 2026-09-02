# EvoGym Technical Reference for Developmental Morphology Integration

Source basis: local checkout at `evogym/`, upstream remote `https://github.com/EvolutionGym/evogym.git`, commit `f63c649`.

External references consulted: EvoGym repository README and examples in the local checkout; upstream EvoGym project page at <https://github.com/EvolutionGym/evogym>; EvoGym paper/appendix links at <https://arxiv.org/abs/2201.09863> and <https://people.csail.mit.edu/jiex/papers/EvoGym/appendix.pdf>.

## Table of Contents

- [A. Installation & Setup](#a-installation--setup)
- [B. Morphology / Design Representation](#b-morphology--design-representation)
- [C. Environment & Simulation API](#c-environment--simulation-api)
- [D. Task / Environment Catalog](#d-task--environment-catalog)
- [E. Controller Training Integration](#e-controller-training-integration)
- [F. Evolutionary Algorithm Integration](#f-evolutionary-algorithm-integration)
- [G. Extensibility Points](#g-extensibility-points)
- [H. File / Repo Map](#h-file--repo-map)
- [I. Open Questions / Risks](#i-open-questions--risks)
- [Summary for Integration Planning](#summary-for-integration-planning)

## A. Installation & Setup

### Package version and Python requirements

- The local checkout declares package name `evogym`, version `2.0.0`, and `requires-python = ">=3.7, <3.11"` in `evogym/pyproject.toml:30-40`.
- Core install dependencies are `gymnasium` and `numpy<2.0.0` in `evogym/pyproject.toml:46-49`.
- The source build backend is Setuptools because the simulator is a native C++/CMake extension (`evogym/pyproject.toml:1-3`).

### Native simulator build

- The native extension is `evogym.simulator_cpp`, built from `evogym/simulator` by `CMakeExtension` / `CMakeBuild` in `evogym/setup.py:12-64`.
- `CMakeBuild.run()` checks for `cmake --version` and raises if CMake is absent (`evogym/setup.py:18-24`).
- Non-Windows builds pass `-DCMAKE_BUILD_TYPE=Release` and build with `-- -j8` (`evogym/setup.py:39-50`).
- Windows requires Visual Studio build tools and CMake >= 3.1.0 according to `evogym/setup.py:26-29` and `evogym/README.md:67-71`.

### Recommended install paths

- Binary package: `pip install evogym --upgrade` per `evogym/README.md:42-48`.
- Source install: clone with submodules, then `pip install -e .` per `evogym/README.md:73-89`.
- Linux system packages: `xorg-dev libglu1-mesa-dev` are listed in `evogym/README.md:57-61` and repeated for source builds at `evogym/README.md:79-83`.
- Example/co-design dependencies are separate from core package metadata. `evogym/requirements.txt:1-5` adds `neat-python`, `GPyOpt`, `stable-baselines3`, `imageio`, and `pygifsicle`.

### Local checkout status / gotchas observed

- The local checkout has not built `evogym.simulator_cpp`; importing environments from source with `python3` fails with `ModuleNotFoundError: No module named 'evogym.simulator_cpp'`. Build/install is required before running simulations.
- The default shell has no `python` executable, only `python3` was available during inspection. EvoGym docs use `python`; local commands may need `python3`.
- The README notes a Conda/Linux OpenGL issue, `libGL error: MESA-LOADER: failed to open iris`, with fix `conda install -c conda-forge libstdcxx-ng` (`evogym/README.md:125-131`).
- Headless usage is supported by creating environments with `render_mode=None`; rendering libraries are initialized only when rendering is requested (`evogym/README.md:172-179`).

## B. Morphology / Design Representation

### Voxel material IDs

The authoritative Python constants are `VOXEL_TYPES` in `evogym/utils.py:8-15`:

| Name | Integer | Meaning |
| --- | ---: | --- |
| `EMPTY` | `0` | No voxel |
| `RIGID` | `1` | Rigid voxel |
| `SOFT` | `2` | Soft passive voxel |
| `H_ACT` | `3` | Horizontal actuator voxel |
| `V_ACT` | `4` | Vertical actuator voxel |
| `FIXED` | `5` | Fixed voxel, typically terrain/static objects |

The C++ enum mirrors this order in `evogym/simulator/SimulatorCPP/main.h:32-34`: `CELL_EMPTY = 0`, then `CELL_RIGID`, `CELL_SOFT`, `CELL_ACT_H`, `CELL_ACT_V`, `CELL_FIXED`.

### Body array shape, dtype, and indexing convention

- User-facing robot bodies are `np.ndarray` matrices. Public docstrings call the body a `(n, m)` array specifying voxel structure in `EvoWorld.add_from_array()` (`evogym/world.py:74-89`) and `WorldObject.from_array()` / `load_from_array()` (`evogym/world.py:286-315`).
- Random sampling accepts `robot_shape: Tuple[int, int]` described as `(h, w)` in `sample_robot()` (`evogym/utils.py:95-104`).
- `sample_robot()` creates bodies with `np.zeros(robot_shape)`, so sampled bodies are float arrays unless converted later (`evogym/utils.py:130-133`). The simulator binding accepts numeric matrices; developmental code should emit integer-valued arrays even if dtype is `float64` or an integer dtype.
- Connections are indexed against `np.flatten(structure)` according to `WorldObject.from_array()` / `load_from_array()` docs (`evogym/world.py:286-315`) and `get_full_connectivity()` docs (`evogym/utils.py:240-249`).
- User-facing row/column convention: row-major `(row, col)` with row 0 as the top row and col 0 as the left column. Evidence: `WorldObject.load_from_array()` stores `self.grid = np.flipud(structure)` (`evogym/world.py:326`) so the internal world grid uses bottom-up y, while `get_structure()` returns `np.flipud(self.grid)` back to the user-facing matrix (`evogym/world.py:465-472`).
- World placement uses physical-grid coordinates `(x, y)` for the bottom-left voxel of the object, starting at 0 (`evogym/world.py:84-89`, `evogym/world.py:450-463`).

### Connectivity representation

- `connections` is a `(2, k)` NumPy array, where each column is a pair of flattened voxel indices into `np.flatten(structure)` (`evogym/world.py:84-89`, `evogym/utils.py:240-249`).
- If `connections` is `None`, `WorldObject.load_from_array()` computes full adjacent connectivity with `get_full_connectivity(structure).T` internally (`evogym/world.py:319-324`).
- `get_full_connectivity(robot)` connects non-empty right and bottom neighbors only, then returns shape `(2, k)` (`evogym/utils.py:240-276`). This encodes each undirected adjacency once.

### Design validation

There is no single high-level `validate_robot()` function in the core package. Validation is distributed:

- `is_connected(robot)` flood-fills non-empty cells and returns `False` for an all-empty robot (`evogym/utils.py:190-222`).
- `has_actuator(robot)` checks for at least one `3` or `4` voxel (`evogym/utils.py:224-238`).
- `sample_robot(robot_shape, pd)` loops until both `is_connected(robot)` and `has_actuator(robot)` hold (`evogym/utils.py:95-138`).
- `WorldObject.load_from_array()` validates only that supplied connection indices refer to non-empty voxels after coordinate flipping; bad connection endpoints raise `ValueError` (`evogym/world.py:350-356`).
- `EvoWorld.add_object()` rejects duplicate object names and overlap with existing world grid cells (`evogym/world.py:95-137`).
- The C++ loader additionally rejects cells outside the world grid and overlapping cells (`evogym/simulator/SimulatorCPP/ObjectCreator.cpp:173-190`, `evogym/simulator/SimulatorCPP/ObjectCreator.cpp:203-214`).
- The C++ loader enforces one connected component after constructing components from the connection graph; if any non-empty cell is in component `>= 1`, it prints `object must be connected` and returns `false` (`evogym/simulator/SimulatorCPP/ObjectCreator.cpp:560-579`).
- `EvoSim._init_world_items()` classifies any object with at least one `H_ACT` or `V_ACT` as a robot (`evogym/sim.py:49-69`). If a Gym task receives a body with no actuator, object name `robot` will not be registered as a robot, and later `get_actuator_indices('robot')` will fail via `_check_valid_robot_name()` (`evogym/sim.py:97-108`, `evogym/sim.py:154-162`).

Required validation for generated 5x5 morphologies should therefore be:

- Shape exactly `(5, 5)`.
- Values restricted to allowed robot material IDs. For evolved robot bodies, examples use domain `{0,1,2,3,4}` and exclude `FIXED` (`evogym/utils.py:95-138`, `evogym/examples/bo/run.py:119-122`, `evogym/examples/cppn_neat/run.py:31-40`).
- At least one non-empty voxel.
- One connected non-empty component under cardinal adjacency.
- At least one actuator (`H_ACT` or `V_ACT`).
- Connections either omitted/`None` or equal to `get_full_connectivity(body)`.

### JSON loading vs in-memory arrays

JSON world/object files:

- `EvoWorld.from_json(file_path)` creates a world and delegates to `add_from_json()` (`evogym/world.py:24-37`).
- `EvoWorld.add_from_json()` expects top-level keys `grid_width`, `grid_height`, and `objects`; each object has `indices`, `types`, and `neighbors` (`evogym/world.py:39-72`).
- `WorldObject.from_json(file_path)` loads an object from a JSON world file and requires that the file contain exactly one object (`evogym/world.py:257-273`).
- `WorldObject.load_from_parsed_json()` converts flattened world indices to `(x, y)` using `Pair(index % grid_size.x, index // grid_size.x)`, computes a bounding box, stores object-local voxels, material grid, and neighbor lists (`evogym/world.py:367-433`).
- Example robot JSON: `evogym/examples/world_data/speed_bot.json`.
- Built-in environment JSONs live under `evogym/evogym/envs/sim_files/*.json` and are packaged by `evogym/pyproject.toml:11-13`.

In-memory arrays:

- `EvoWorld.add_from_array(name, structure, x, y, connections=None)` creates a `WorldObject` from a NumPy body and inserts it into the world (`evogym/world.py:74-93`).
- Every built-in Gym task follows this convention: load a task world from JSON, then call `self.world.add_from_array('robot', body, ..., connections=connections)`. Walker example: `evogym/envs/walk.py:25-30`.

Saving:

- The current core package exposes load-from-JSON and load-from-array APIs but no symmetric high-level `to_json()` writer in `evogym/world.py`.
- Example experiments save evaluated bodies and connectivity with `np.savez(...)`, not JSON: PPO saves `robot.get_structure()` and `robot.get_connections()` in `evogym/examples/run_ppo.py:46-49`; GA saves each `Structure.body` and `Structure.connections` in `evogym/examples/ga/run.py:143-147`; CPPN-NEAT does the same in `evogym/examples/cppn_neat/run.py:46-50`.

## C. Environment & Simulation API

### Instantiating Walker-v0 with a custom voxel grid

- Environment registration is done in `evogym/envs/__init__.py`; `Walker-v0` maps to `evogym.envs.walk:WalkingFlat` and has `max_episode_steps=500` (`evogym/envs/__init__.py:16-20`).
- Importing `evogym.envs` is required to register environments, as shown in the README installation test (`evogym/README.md:101-110`).
- `WalkingFlat.__init__(body, connections=None, render_mode=None, render_options=None)` loads `Walker-v0.json`, adds `body` as object name `robot` at `(x=1, y=1)`, and initializes `BenchmarkBase` (`evogym/envs/walk.py:15-30`).
- The supported hook is the `body` argument and optional `connections` argument passed to `gymnasium.make()`, usually through direct kwargs or Stable-Baselines3 `env_kwargs`.

### Walker-v0 observation, action, reset, and step

For `Walker-v0` / `WalkingFlat`:

- Number of actuators is `self.get_actuator_indices('robot').size` (`evogym/envs/walk.py:32-36`).
- Number of robot point-position scalars is `self.object_pos_at_time(self.get_time(), "robot").size` (`evogym/envs/walk.py:33-34`).
- `action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators,), dtype=float)` (`evogym/envs/walk.py:36`).
- `observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2 + num_robot_points,), dtype=float)` (`evogym/envs/walk.py:37`).
- `reset()` returns `(obs, {})`, where `obs = concat(get_vel_com_obs("robot"), get_relative_pos_obs("robot"))` (`evogym/envs/walk.py:74-84`).
- `step(action)` captures pre/post robot positions, calls the base step as `super().step({'robot': action})`, returns the same observation structure, and rewards forward x COM displacement (`evogym/envs/walk.py:39-72`).
- If the C++ simulator reports instability/self-collision, `done` is true and Walker subtracts `3.0` reward (`evogym/envs/walk.py:61-64`).
- If robot COM x exceeds `99 * VOXEL_SIZE`, Walker terminates and adds `1.0` reward (`evogym/envs/walk.py:66-69`).
- `BenchmarkBase.VOXEL_SIZE` is `0.1` (`evogym/envs/base.py:381-383`), matching C++ cell size `Vector2d(0.1, 0.1)` in `evogym/simulator/SimulatorCPP/ObjectCreator.h:31`.

### Base simulation loop

- `EvoGymBase.step(action_dict)` expects a dictionary mapping robot names to action arrays, clips each action to `[0.6, 1.6]`, sends it to `EvoSim.set_action()`, then calls `self._sim.step()` (`evogym/envs/base.py:43-63`).
- `EvoGymBase.reset()` calls `self._sim.reset()` (`evogym/envs/base.py:65-69`).
- `EvoSim.step()` in C++ performs 30 physics updates per Gym step and increments simulator time once (`evogym/simulator/SimulatorCPP/Sim.cpp:12-21`, `evogym/simulator/SimulatorCPP/Sim.cpp:103-112`).
- The C++ `Environment::step()` runs collision resolution, actuator updates, RK4 integration, edge constraints, and returns whether any robot is self-colliding (`evogym/simulator/SimulatorCPP/Environment.cpp:166-207`).

### Action vector to actuator mapping

- `EvoSim._init_world_items()` stores actuator indices from `self.get_indices_of_actuators(obj.name).flatten()` after `read_robot_from_array()` (`evogym/sim.py:87-92`).
- `EvoSim.get_actuator_indices(robot_name)` returns that stored array (`evogym/sim.py:97-108`).
- `EvoSim.set_action(robot_name, action)` verifies that the flattened action shape equals the actuator-index array shape, then stacks `(actuator_index, action_value)` pairs before calling C++ `set_action()` (`evogym/sim.py:123-140`).
- C++ `Robot::init()` scans the robot `boxels` vector and records `grid_index` for boxels whose type is `CELL_ACT_H` or `CELL_ACT_V` (`evogym/simulator/SimulatorCPP/Robot.cpp:9-22`).
- C++ `Environment::set_robot_action()` maps each row’s grid index to an actuator boxel and passes the actuator/value lists to `PhysicsEngine::update_actuator_goals()` (`evogym/simulator/SimulatorCPP/Environment.cpp:220-236`).
- `PhysicsEngine::update_actuator_goals()` applies horizontal actuator values to top/bottom edges and vertical actuator values to left/right edges (`evogym/simulator/SimulatorCPP/PhysicsEngine.cpp:38-89`).

Practical rule: after environment construction, call `env.unwrapped.get_actuator_indices('robot')` to label the action slots. For standard user-facing arrays, these indices correspond to flattened local robot grid indices; action slot `i` controls the actuator whose flattened grid index is `get_actuator_indices('robot')[i]`.

## D. Task / Environment Catalog

Authoritative registration list: `evogym/envs/__init__.py:16-209`. The README states EvoGym includes 32 locomotion/manipulation tasks (`evogym/README.md:6`).

| Env ID | Class | File | Max steps | Objective / reward summary |
| --- | --- | --- | ---: | --- |
| `Walker-v0` | `WalkingFlat` | `evogym/envs/walk.py` | 500 | Walk right on flat terrain; reward is robot COM `+x` displacement, plus terminal bonus at terrain end (`walk.py:56-69`). |
| `BridgeWalker-v0` | `SoftBridge` | `evogym/envs/walk.py` | 500 | Walk right on a soft bridge; reward is COM `+x`, plus end bonus (`walk.py:128-141`). |
| `CaveCrawler-v0` | `Duck` | `evogym/envs/walk.py` | 1000 | Move right through low ceiling/terrain; reward is COM `+x`, plus end bonus (`walk.py:203-216`). |
| `Jumper-v0` | `StationaryJump` | `evogym/envs/jump.py` | 500 | Jump upward in place; reward is `10 * y` displacement minus `5 * abs(x displacement)` (`jump.py:56-66`). |
| `Flipper-v0` | `Flipping` | `evogym/envs/flip.py` | 600 | Rotate counter-clockwise; reward is change in unwrapped orientation (`flip.py:59-85`). |
| `Balancer-v0` | `Balance` | `evogym/envs/balance.py` | 600 | Balance on a pole; reward is reduction in distance to target COM coordinates (`balance.py:47-79`). |
| `Balancer-v1` | `BalanceJump` | `evogym/envs/balance.py` | 600 | Balance/jump toward a higher target; reward is reduction in distance to target COM coordinates (`balance.py:129-162`). |
| `UpStepper-v0` | `StepsUp` | `evogym/envs/traverse.py` | 600 | Move right up stairs; reward is COM `+x`, bonus at end, penalty for tipping (`traverse.py:21-32`, `traverse.py:92-110`). |
| `DownStepper-v0` | `StepsDown` | `evogym/envs/traverse.py` | 500 | Move right down stairs; reward is COM `+x`, bonus at end, penalty for tipping (`traverse.py:157-175`). |
| `ObstacleTraverser-v0` | `WalkingBumpy` | `evogym/envs/traverse.py` | 1000 | Move right over increasingly bumpy terrain; reward is COM `+x`, bonus at end, penalty for tipping (`traverse.py:223-241`). |
| `ObstacleTraverser-v1` | `WalkingBumpy2` | `evogym/envs/traverse.py` | 1000 | Move right over harder bumpy terrain; reward is COM `+x`, bonus at end (`traverse.py:289-304`). |
| `Hurdler-v0` | `VerticalBarrier` | `evogym/envs/traverse.py` | 1000 | Move right across tall obstacles; reward is COM `+x`, penalty for tipping (`traverse.py:352-365`). |
| `GapJumper-v0` | `Gaps` | `evogym/envs/traverse.py` | 1000 | Move right across separated platforms; reward is COM `+x`, penalty for falling (`traverse.py:500-516`). |
| `PlatformJumper-v0` | `FloatingPlatform` | `evogym/envs/traverse.py` | 1000 | Move right across floating platforms at varied heights; reward is COM `+x`, penalty for falling/tipping (`traverse.py:415-434`). |
| `Traverser-v0` | `BlockSoup` | `evogym/envs/traverse.py` | 600 | Move right through a pit of blocks; reward is COM `+x`, bonus at end (`traverse.py:610-625`). |
| `Lifter-v0` | `LiftSmallRect` | `evogym/envs/manipulate.py` | 300 | Lift a package from a hole; reward is package `+y`, x-position shaping, and robot-fall penalty (`manipulate.py:875-916`). |
| `Carrier-v0` | `CarrySmallRect` | `evogym/envs/manipulate.py` | 500 | Carry a package right; reward is robot/package `+x`, penalty if package drops below threshold, plus end bonus (`manipulate.py:112-162`). |
| `Carrier-v1` | `CarrySmallRectToTable` | `evogym/envs/manipulate.py` | 1000 | Carry package toward table/goal; reward shapes robot/package x-goal progress and package height (`manipulate.py:199-244`). |
| `Pusher-v0` | `PushSmallRect` | `evogym/envs/manipulate.py` | 500 | Push package right; reward uses `PackageBase.get_reward()` for package/robot progress and separation/drop penalties, plus end bonus (`manipulate.py:40-58`, `manipulate.py:277-310`). |
| `Pusher-v1` | `PushSmallRectOnOppositeSide` | `evogym/envs/manipulate.py` | 600 | Push package from opposite side; same base reward style, plus end bonus (`manipulate.py:343-376`). |
| `BeamToppler-v0` | `ToppleBeam` | `evogym/envs/manipulate.py` | 1000 | Approach and topple a beam; reward for reducing robot-beam x distance, beam movement, and beam falling (`manipulate.py:646-704`). |
| `BeamSlider-v0` | `SlideBeam` | `evogym/envs/manipulate.py` | 1000 | Approach and slide a beam; reward for reducing robot-beam distance and beam `+x` movement (`manipulate.py:765-819`). |
| `Thrower-v0` | `ThrowSmallRect` | `evogym/envs/manipulate.py` | 300 | Throw package right while penalizing robot drift from x=0 (`manipulate.py:415-452`). |
| `Catcher-v0` | `CatchSmallRect` | `evogym/envs/manipulate.py` | 400 | Catch/follow falling package; reward reduces robot-package x separation and penalizes package drop (`manipulate.py:535-571`). |
| `AreaMaximizer-v0` | `MaximizeShape` | `evogym/envs/change_shape.py` | 600 | Maximize convex-hull area; reward is positive area change scaled by 10 (`change_shape.py:136-148`). |
| `AreaMinimizer-v0` | `MinimizeShape` | `evogym/envs/change_shape.py` | 600 | Minimize convex-hull area; reward is negative area change scaled by 10 (`change_shape.py:201-213`). |
| `WingspanMazimizer-v0` | `MaximizeXShape` | `evogym/envs/change_shape.py` | 600 | Maximize x-span; reward is increase in x extent (`change_shape.py:266-279`). Note spelling is `Mazimizer` in registry (`envs/__init__.py:174-178`). |
| `HeightMaximizer-v0` | `MaximizeYShape` | `evogym/envs/change_shape.py` | 500 | Maximize y-span; reward is increase in y extent (`change_shape.py:332-345`). |
| `Climber-v0` | `Climb0` | `evogym/envs/climb.py` | 400 | Climb vertical channel; reward is robot COM `+y`, plus end bonus (`climb.py:78-92`). |
| `Climber-v1` | `Climb1` | `evogym/envs/climb.py` | 600 | Climb mixed-material vertical channel; reward is robot COM `+y`, plus end bonus (`climb.py:136-150`). |
| `Climber-v2` | `Climb2` | `evogym/envs/climb.py` | 1000 | Climb/navigate pipe; reward is COM `+y` plus `0.2 * +x` (`climb.py:197-207`). |
| `BidirectionalWalker-v0` | `BiWalk` | `evogym/envs/multi_goal.py` | 1000 | Walk toward randomized x-goals; reward is reduction in distance to current goal, plus final completion bonus (`multi_goal.py:71-96`, `multi_goal.py:173-184`). |

## E. Controller Training Integration

### EvoGym's PPO example stack

- PPO is from `stable_baselines3.PPO`, imported in `evogym/examples/ppo/run.py:1-5` and `evogym/examples/run_ppo.py:1-8`.
- Vectorized environments are created through `stable_baselines3.common.env_util.make_vec_env` (`evogym/examples/ppo/run.py:4-5`).
- Morphology is passed by `env_kwargs={'body': body, 'connections': connections}` in `run_ppo()` (`evogym/examples/ppo/run.py:23-27`).
- The policy is `"MlpPolicy"` with hyperparameters from CLI args (`evogym/examples/ppo/run.py:42-57`).
- Training calls `model.learn(total_timesteps=args.total_timesteps, callback=callback, log_interval=args.log_interval)` (`evogym/examples/ppo/run.py:58-62`).
- The custom `EvalCallback` periodically calls `eval_policy()` and saves the best model (`evogym/examples/ppo/callback.py:92-115`).
- `run_ppo.py` loads a robot from JSON with `WorldObject.from_json()`, saves body/connections with `np.savez`, trains, then loads/evaluates the best controller (`evogym/examples/run_ppo.py:46-79`).

### Minimal training loop, as an API sequence

1. Register environments by importing `evogym.envs`.
2. Produce or load a valid `body` array and optional `connections` array.
3. Create vectorized environment with ID such as `Walker-v0` and `env_kwargs` containing `body` and `connections`.
4. Instantiate `stable_baselines3.PPO` with `"MlpPolicy"` and the vectorized environment.
5. Call `PPO.learn(...)` for the desired timestep budget.
6. Evaluate with new environments using the same `body` and `connections`.
7. Save controller and body/connectivity together so the controller can be replayed only against the morphology it was trained on.

### Training budget

- The example README gives a quick PPO command for Walker using `--total-timesteps 100000` and says this is customizable (`evogym/examples/README.md:8-16`).
- The same README recommends `total-timesteps > 1e6` for a serious training run (`evogym/examples/README.md:16`).
- The PPO arg parser default is `--total-timesteps 1e6` (`evogym/examples/ppo/args.py:43-45`) and default `--eval-interval 1e5` (`evogym/examples/ppo/args.py:58-60`).
- Current example co-design commands also use `--total-timesteps 100000` as testing/demo budgets and explicitly warn they will not produce task-performant robots (`evogym/examples/README.md:34-42`).
- The current repo points readers to Appendix D of the original paper for exact evaluation parameters (`evogym/examples/README.md:42`), but the local checkout does not include those extracted hyperparameter tables. Treat paper-exact budgets as an external reproducibility item to verify before claiming reproduction.

## F. Evolutionary Algorithm Integration

EvoGym core does not expose a reusable GA/ES API in the package namespace. It ships example co-design scripts under `evogym/examples/`:

- Direct-grid genetic algorithm: `evogym/examples/run_ga.py`, `evogym/examples/ga/run.py`, `evogym/examples/utils/algo_utils.py`.
- Bayesian optimization over categorical direct-grid genomes: `evogym/examples/run_bo.py`, `evogym/examples/bo/run.py`, `evogym/examples/bo/optimizer.py`.
- CPPN-NEAT indirect encoding: `evogym/examples/run_cppn_neat.py`, `evogym/examples/cppn_neat/run.py`, `evogym/examples/cppn_neat/population.py`, `evogym/examples/cppn_neat/neat.cfg`.

### Direct GA example

- Genome/phenotype format is a direct `body` matrix plus `connections`, wrapped by `Structure(body, connections, label)` (`evogym/examples/utils/algo_utils.py:5-33`).
- Initial population is generated by `sample_robot(structure_shape)` and deduplicated by `hashable(body)` (`evogym/examples/ga/run.py:86-103`).
- Fitness is just `Structure.reward` (`evogym/examples/utils/algo_utils.py:19-27`).
- Each non-survivor is evaluated by calling `run_ppo(args, structure.body, env_name, save_path_controller, label, structure.connections)` (`evogym/examples/ga/run.py:148-170`).
- Survivors are sorted by fitness and copied forward without retraining; prior controllers are copied from the previous generation (`evogym/examples/ga/run.py:176-208`).
- The implemented reproduction step selects one survivor parent and mutates it; despite the comment header "CROSSOVER AND MUTATION", no crossover operator is implemented in `ga/run.py:200-223`.
- Mutation loops over cells with probability `mutation_rate`; mutated cells are resampled from a 5-category distribution over `{EMPTY,RIGID,SOFT,H_ACT,V_ACT}` with `EMPTY` probability set to `0.6`, then accepted only if connected and actuator-bearing (`evogym/examples/utils/algo_utils.py:46-64`).

### Bayesian optimization example

- Genome is a 1D categorical vector with domain `(0,1,2,3,4)` and dimensionality `np.prod(structure_shape)` (`evogym/examples/bo/run.py:119-122`).
- `get_robot_from_genome()` reshapes the vector to `structure_shape` (`evogym/examples/bo/run.py:16-22`).
- Invalid robots return cost `10`; valid robots save body/connectivity and call `run_ppo()`, with cost `-fitness` (`evogym/examples/bo/run.py:24-42`).
- Constraint checking uses `is_connected(robot) and has_actuator(robot)` (`evogym/examples/bo/run.py:44-50`).

### CPPN-NEAT example

- CPPN inputs are coordinate tensors `x`, `y`, and distance from center `d` (`evogym/examples/cppn_neat/run.py:24-29`).
- CPPN outputs are named `empty`, `rigid`, `soft`, `hori`, and `vert`; `argmax` over the five outputs gives material IDs `0..4`, reshaped to `structure_shape` (`evogym/examples/cppn_neat/run.py:31-40`).
- Validity requires connectivity, at least one actuator, and no duplicate structure hash (`evogym/examples/cppn_neat/run.py:57-66`).
- Valid genomes are evaluated by computing `get_full_connectivity(robot)`, saving `.npz`, and calling `run_ppo()` (`evogym/examples/cppn_neat/run.py:42-55`).
- NEAT config uses `num_inputs=3`, `num_hidden=1`, `num_outputs=5`, feed-forward networks, sigmoid activation, connection/node add/delete probabilities, and weight/bias mutation settings (`evogym/examples/cppn_neat/neat.cfg:10-50`).
- `Population.run()` replaces invalid genomes before evaluating fitness by repeatedly creating new genomes until the population is valid (`evogym/examples/cppn_neat/population.py:27-50`).

## G. Extensibility Points

### Where to plug in custom developmental encodings

There is no formal callback named "genome to morphology" in the core package. The convention in all examples is:

1. Outer-loop optimizer owns the genome.
2. Outer-loop code decodes/generates a `body` matrix of material IDs.
3. Outer-loop code validates `body` using `is_connected()` and `has_actuator()`.
4. Outer-loop code computes `connections = get_full_connectivity(body)` unless custom joint topology is desired.
5. PPO/simulation receives `body` and `connections` through `gym.make(..., body=body, connections=connections)` or SB3 `env_kwargs`.

Source templates:

- Direct GA insertion point: replace `sample_robot()` / `mutate()` output with `genome -> body -> connections` before `run_ppo()` in `evogym/examples/ga/run.py:92-103` and `evogym/examples/ga/run.py:210-223`.
- BO insertion point: replace `get_robot_from_genome()` in `evogym/examples/bo/run.py:16-22`.
- CPPN-NEAT insertion point: replace or adapt `get_robot_from_genome()` in `evogym/examples/cppn_neat/run.py:31-40`.
- PPO-only fixed morphology path: replace `WorldObject.from_json(args.robot_path)` with a generated body before the `run_ppo()` call in `evogym/examples/run_ppo.py:46-59`.

### Existing indirect-encoding example

`evogym/examples/cppn_neat/run.py` is the closest structural template for developmental systems:

- It keeps genotype/network separate from final body.
- It decodes the genotype to a final voxel grid before PPO starts.
- It validates the final grid before running PPO.
- It evolves the generator/genotype and never mutates the trained controller as part of the genome.

This matches the capstone plan better than the direct GA. A symbolic IF-THEN developmental rule system or NCA can follow the same outer-loop shape: `genome -> developmental rollout -> final 5x5 body -> validation -> PPO`.

### Strict 5x5 constraints / small-grid gotchas

- EvoGym examples default to 5x5 structures in `run_ga.py`, `run_bo.py`, and `run_cppn_neat.py` (`evogym/examples/run_ga.py:12-15`, `evogym/examples/run_bo.py:12-15`, `evogym/examples/run_cppn_neat.py:12-15`).
- `sample_robot((5,5))` is shown in the README installation test (`evogym/README.md:101-123`), so 5x5 is a first-class supported scale.
- For a 5x5 all-empty or disconnected body, Python validation fails (`is_connected()` returns `False` for all-empty; disconnected components return `False`) and C++ loader also rejects multiple connected components.
- A single actuator voxel is connected and actuator-bearing, but `get_full_connectivity()` returns no connections for a one-voxel robot (`evogym/utils.py:240-276`). This is technically valid by Python helpers, but may be a poor or unstable morphology for training; consider project-level minimum occupied-voxel or minimum-connection constraints.
- `WorldObject.load_from_array()` does not explicitly reject material IDs outside `0..5`. Unknown IDs can pass Python loading but will not match expected C++ material cases in `ObjectCreator.cpp`; developmental code should clamp or reject invalid material IDs before environment creation.
- `FIXED` voxels can appear in arrays, but the co-design examples use only IDs `0..4` for robot bodies. Treat `FIXED` as environment/static material unless deliberately testing anchored robot designs.
- The registered task ID is misspelled `WingspanMazimizer-v0`; `WingspanMaximizer-v0` is not registered in this checkout (`evogym/envs/__init__.py:174-178`).
- The current examples use Gymnasium-style environments (`reset()` returns `(obs, info)` and env `step()` returns five values), but SB3 VecEnv wrappers still expose four-return vectorized `step()` in the example evaluation code (`evogym/examples/ppo/eval.py:30-41`). Version pinning matters.

## H. File / Repo Map

```text
evogym/
  README.md
    Install instructions, headless rendering notes, example task count, pointers to tutorials/examples.
  pyproject.toml
    Package metadata, Python range, core dependencies, native package-data config.
  setup.py
    CMake build hook for evogym.simulator_cpp.
  requirements.txt
    Extra dependencies for examples: SB3, NEAT, GPyOpt, GIF tools.
  evogym/
    __init__.py
      Exports EvoWorld, WorldObject, EvoSim, EvoViewer, and utils constants/helpers.
    utils.py
      Voxel constants, baseline env names, random robot sampler, connectivity and actuator checks.
    world.py
      EvoWorld and WorldObject; JSON and array loading; object placement; connection conversion.
    sim.py
      Python wrapper around native Sim; world-to-simulator loading; actuator indexing/action dispatch.
    envs/
      __init__.py
        Gymnasium registration for built-in task IDs.
      base.py
        EvoGymBase and BenchmarkBase; Gym step/reset/render helpers and observation utilities.
      walk.py
        Walker-v0, BridgeWalker-v0, CaveCrawler-v0.
      traverse.py
        Step, obstacle, platform, gap, and block traversal tasks.
      climb.py
        Climber-v0/v1/v2 tasks.
      manipulate.py
        Carrier, pusher, thrower, catcher, beam, and lifter tasks.
      change_shape.py
        Area/wingspan/height shape-change tasks.
      jump.py
        Jumper-v0.
      flip.py
        Flipper-v0.
      balance.py
        Balancer-v0/v1.
      multi_goal.py
        BidirectionalWalker-v0 and goal sequencing.
      sim_files/*.json
        Built-in terrain/package/object world definitions.
    simulator/SimulatorCPP/
      main.h
        C++ cell type enum matching Python voxel IDs.
      ObjectCreator.cpp/.h
        Converts grids/connections into simulator objects; validates world bounds, overlap, connected components.
      Robot.cpp/.h
        Collects actuator grid indices and maps actions to actuator boxels.
      Environment.cpp/.h
        Simulation step, object registry, robot action dispatch, snapshots.
      PhysicsEngine.cpp/.h
        Physics integration and actuator target updates.
  examples/
    README.md
      How to run PPO, GA, BO, CPPN-NEAT; demo and serious timestep guidance.
    run_ppo.py
      Fixed-robot PPO entry point using JSON-loaded robot.
    ppo/
      run.py
        SB3 PPO construction/training entry point.
      args.py
        PPO hyperparameter CLI defaults.
      callback.py
        Periodic evaluation and best-controller saving.
      eval.py
        Vectorized evaluation loop.
    run_ga.py, ga/run.py
      Direct-grid GA example wrapping PPO evaluations.
    run_bo.py, bo/run.py, bo/optimizer.py
      Categorical Bayesian optimization example wrapping PPO evaluations.
    run_cppn_neat.py, cppn_neat/run.py, cppn_neat/population.py, cppn_neat/neat.cfg
      CPPN-NEAT indirect encoding example wrapping PPO evaluations.
    world_data/*.json
      Example standalone robot/world JSONs.
  tests/
    screen_free/test_utils.py
      Unit tests for voxel constants, connectivity, actuators, full connectivity, robot sampling.
    screen_free/test_baseline_envs.py
      Smoke test for registered environments with sampled 5x5 robots.
```

## I. Open Questions / Risks

- The local source tree is not runnable until the native extension is built. Any runtime confirmation of observation/action dimensions should wait until `pip install -e .` succeeds.
- The current examples are explicitly modified for usability and warn that exact paper reproduction should use the original release (`evogym/examples/README.md:5-6`). Do not mix current example defaults with paper claims without pinning a release.
- Paper-exact PPO budgets from Appendix D were not extractable with local tools. The current repo gives demo `100000`, serious `>1e6`, and parser default `1e6`; exact original experimental budgets remain a verification item.
- Core constructors do not comprehensively validate value ranges or project-level design constraints. Developmental encodings should run their own validator before calling `gym.make()`.
- `FIXED` is a valid material type globally but not used by co-design examples as a robot material. Decide whether generated morphologies may use it; default recommendation is no.
- Action ordering should be treated as discoverable via `get_actuator_indices('robot')`, not hard-coded. This avoids mistakes if array memory order or non-square grids enter later experiments.
- `connections=None` gives full cardinal adjacency. Any developmental encoding that wants nonstandard joint topology must produce a valid `(2,k)` connection matrix over non-empty flattened cells.
- Some examples use broad `except:` blocks and interactive overwrite prompts; production capstone runs should wrap these scripts or reimplement orchestration carefully.
- `run_ppo()` accepts `args.n_envs`, but currently creates `make_vec_env(..., n_envs=1)` in `evogym/examples/ppo/run.py:23-27`; parallelism is handled elsewhere for robot evaluations. This may surprise users expecting PPO vectorization from `--n-envs`.
- `Catcher-v0.reset()` reconstructs the world by calling `random_init()` after base reset (`evogym/envs/manipulate.py:577-608`); it is less deterministic than most tasks unless random seeds are controlled carefully.

## Summary for Integration Planning

- Our developmental genome should not be passed to EvoGym directly.
- The developmental decoder must output a final `np.ndarray` body before PPO starts.
- For the planned fixed design space, body shape should be exactly `(5, 5)`.
- Body values should be integer material IDs: `0` empty, `1` rigid, `2` soft, `3` horizontal actuator, `4` vertical actuator.
- Avoid `5` fixed for robot morphologies unless we intentionally want anchored material.
- Validate at least one non-empty voxel, cardinal connectivity, and at least one actuator.
- Use `evogym.is_connected(body)` and `evogym.has_actuator(body)` as baseline checks.
- Use `evogym.get_full_connectivity(body)` unless experimenting with custom connectivity.
- Instantiate tasks by passing `body` and `connections` into `gymnasium.make('Walker-v0', ...)` after importing `evogym.envs`.
- PPO action dimension equals number of actuator voxels, not number of occupied voxels.
- Label action outputs by `env.unwrapped.get_actuator_indices('robot')`.
- For Walker-v0, observation is COM velocity plus relative point positions; reward is forward COM displacement.
- The closest repo template for indirect developmental encodings is `examples/cppn_neat/run.py`.
- For capstone evolution, replace `get_robot_from_genome()` with `rule_genome -> final body` or `nca_weights -> final body`.
- Keep controller training inside the evaluation function: generated body -> PPO train -> evaluation reward -> genome fitness.
