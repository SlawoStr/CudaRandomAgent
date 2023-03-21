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

|Event|Action|  
|---|---|  
|**M**|Switch between CPU/GPU|  
|**1**|Switch agent type to SEEKER|  
|**2**|Switch agent type to ARRIVER|  
|**3**|Switch agent type to WANDERER|  
|**Mouse Left**|Spawn 1 agent|   
|**R**|Spawn maximum number of agents|   

# Performance  

|Processor Type \ Agent Number|100'000|1'000'000|5'000'000|10'000'000|  
|---|---|---|---|---|
|**CPU**|0.5ms|38ms|190ms|375ms|  
|**GPU**|0.5ms|1.8ms|8ms|16ms|  

# Visualisation  

- Seeker   

![Seeker Animation](https://github.com/SlawoStr/CudaRandomAgent/blob/master/Img/Seeker.gif)

- Arriver 

![Arriver Animation](https://github.com/SlawoStr/CudaRandomAgent/blob/master/Img/Arriver.gif)

- Wanderer 

![Wanderer Animation](https://github.com/SlawoStr/CudaRandomAgent/blob/master/Img/Wanderer.gif)




