#pragma once
#include <thrust/device_vector.h>
#include "EntityManager.h"
#include "OpenGL/Shader.h"
#include "OpenGL/VAO.h"
#include "OpenGL/VBO.h"
#include "Utilities/CudaRNG.cuh"
#include "cuda_runtime.h"

class GPUEntityManager : public EntityManager
{
public:
	GPUEntityManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, int blockNumber);

	~GPUEntityManager();


protected:
	int								m_threadNumber;
	int								m_blockNumber;
	VBO								m_agentVBO;
	VBO								m_movementVBO;
	VAO								m_agentVAO;
	VAO								m_movementVAO;
	Shader							m_triangleShader;
	Shader							m_lineShader;
	curandState*					m_state;
	struct cudaGraphicsResource*	m_cudaAgentResource;
	struct cudaGraphicsResource*	m_cudaMovementResource;
};

