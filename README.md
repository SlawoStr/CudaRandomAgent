# CudaRandomAgent

Implementation of three different autonomous agents. 

- Seeker - Moves towards the target (Mouse) and tries to pierce it   
- Arriver - Moves toward the target (Mouse, then slows down and stops on it    
- Wanderer - Randomly moves around the map 

# Implementation

- Sequential/Mulithreaded version on CPU      
- GPU Accelerated using Nvidia CUDA technology with openGL interoperability

# Settings

Parameters of agents can be changed via configuration files in resources folder.

# General Controls

|Event|Action|
|---|---|
|**Mouse Middle**|Panning|
|**Mouse Wheel**|Zooming In/Out|
|**Esc**|Close window|
|**P**|Pause/Unpause|


# Simulation Controls

|**M**|Switch between CPU/GPU|
|**1**|Switch agent type to SEEKER|
|**2**|Switch agent type to ARRIVER|
|**3**|Switch agent type to WANDERER|
|**Mouse Left**|Spawn 1 agent|
|**R**|Spawn maximum number of agents|




