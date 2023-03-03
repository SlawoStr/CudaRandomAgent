#pragma once
#include <thrust/device_vector.h>
#include "EntityManager.h"
#include "OpenGL/Shader.h"
#include "OpenGL/VAO.h"
#include "OpenGL/VBO.h"
#include "Utilities/CudaRNG.cuh"
#include "cuda_runtime.h"

/// <summary>
/// Class to be inherited by other cuda agent managers
/// </summary>
class GPUEntityManager : public EntityManager
{
public:
	GPUEntityManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, int blockNumber);

	~GPUEntityManager();

protected:
	int								m_threadNumber;				//!< Number of threads per block
	int								m_blockNumber;				//!< Number of blocks
	VBO								m_agentVBO;					//!< Agent VBO data
	VBO								m_movementVBO;				//!< Agent vector movement VBO data
	VAO								m_agentVAO;					//!< VAO for agent
	VAO								m_movementVAO;				//!< VAO for agent movement
	Shader							m_triangleShader;			//!< Shader for triangles(agents body)
	Shader							m_lineShader;				//!< Shader for line(movement vector)
	curandState*					m_state;					//!< Random states for random number generator
	struct cudaGraphicsResource*	m_cudaAgentResource;		//!< Cuda resources for openGL data interop (agents)
	struct cudaGraphicsResource*	m_cudaMovementResource;		//!< Cuda resources for openGL data interop	(movement vector)
};

