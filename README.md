# CudaRandomAgent

Implementation of three types of agents:

- Seeker - Moves towards the target and tries to pierce it   
- Arriver - Moves toward the target, then slows down and stops on it    
- Wanderer - Randomly moves around the map 

Two implementation
- Sequential/Mulithreaded version on CPU      
- GPU Accelerated using Nvidia CUDA technology with direct contact with openGL to avoid CPU-GPU transfer data.   

Agents settings, simulation bound, agent type, processor type can be set by Config file in Resources   
