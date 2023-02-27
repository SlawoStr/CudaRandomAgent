#include "GPUEntityManager.cuh"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "Utilities/CudaError.h"
#include <SFML/Graphics.hpp>

GPUEntityManager::GPUEntityManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, int blockNumber)
	: EntityManager(maxSpeed,maxForce,maxAgentNumber,drawMode),m_threadNumber(threadNumber),m_blockNumber(blockNumber), m_triangleShader("Resources/default.vert","Resources/default.frag"),m_lineShader("Resources/defaultLine.vert", "Resources/defaultLine.frag")
{
	m_state = generateRandomStates(m_blockNumber, m_threadNumber, static_cast<unsigned>(time(0)));
	// Set Up agents buffers
	m_agentVBO.create(m_maxAgentNumber * 3 * sizeof(float3));
	m_agentVAO.Bind();
	m_agentVAO.LinkAttrib(m_agentVBO, 0, 3, GL_FLOAT, 3 * sizeof(float), (void*)0);
	m_agentVAO.Unbind();
	m_agentVBO.Unbind();
	// Set up movement buffers
	m_movementVBO.create(m_maxAgentNumber * 2 * sizeof(float2));
	m_movementVAO.Bind();
	m_movementVAO.LinkAttrib(m_movementVBO, 0, 2, GL_FLOAT, 2 * sizeof(float), (void*)0);
	m_movementVAO.Unbind();
	m_movementVBO.Unbind();

	// Set up Cuda interop
	cudaSetDevice(0);
	cudaGraphicsGLRegisterBuffer(&m_cudaAgentResource, m_agentVBO.ID, cudaGraphicsMapFlagsNone);
	cudaGraphicsGLRegisterBuffer(&m_cudaMovementResource, m_movementVBO.ID, cudaGraphicsMapFlagsNone);
}

GPUEntityManager::~GPUEntityManager()
{
	cudaFree(m_state);
}