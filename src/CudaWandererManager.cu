#include "CudaWandererManager.cuh"
#include "Utilities/RNGGenerator.h"
#include "Utilities/CudaError.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "device_launch_parameters.h"

namespace
{
	/// <summary>
	/// Convert degrees to radians
	/// </summary>
	/// <param name="a">Degree number</param>
	/// <returns>Radians</returns>
	__device__ float radians(float a)
	{
		return 0.017453292 * a;
	}
}

////////////////////////////////////////////////////////////
/*
__global__ void updateAgent(float3* verticle, float2* lineVerticle, Wanderer* agents, int taskSize, float maxSpeed, float maxForce, float wanderDistance, float wanderRadius, curandState* state, float2 simulationBound)
{
	curandState localState = state[threadIdx.x + blockDim.x * blockIdx.x];
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < taskSize; i += blockDim.x * gridDim.x)
	{
		agents[i].wander(maxSpeed, maxForce, wanderDistance, wanderRadius, randomFloat(localState, 0.0f, 6.2831f), simulationBound);
		agents[i].update(maxSpeed);

		float angle = agents[i].getAngle();
		float2 position = agents[i].getPosition();

		verticle[i * 3].x = position.x + 10 * cos(angle);
		verticle[i * 3].y = position.y + 10 * sin(angle);

		verticle[i * 3 + 1].x = position.x + 6 * cos(angle - radians(90));
		verticle[i * 3 + 1].y = position.y + 6 * sin(angle - radians(90));

		verticle[i * 3 + 2].x = position.x + 6 * cos(angle + radians(90));
		verticle[i * 3 + 2].y = position.y + 6 * sin(angle + radians(90));

		lineVerticle[i * 2].x = position.x;
		lineVerticle[i * 2].y = position.y;

		lineVerticle[i * 2 + 1].x = position.x + 30 * cos(angle);
		lineVerticle[i * 2 + 1].y = position.y + 30 * sin(angle);
	}
	state[threadIdx.x + blockDim.x * blockIdx.x] = localState;
}
*/
__global__ void updateAgent(float3* verticle, float2* lineVerticle, Wanderer* agents, int taskSize, float maxSpeed, float maxForce, float wanderDistance, float wanderRadius, curandState* state, float2 simulationBound)
{
	curandState localState = state[threadIdx.x + blockDim.x * blockIdx.x];
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < taskSize; i += blockDim.x * gridDim.x)
	{
		agents[i].wander(maxSpeed, maxForce, wanderDistance, wanderRadius, randomFloat(localState, 0.0f, 6.2831f), simulationBound);
		agents[i].update(maxSpeed);
		
		float angle = agents[i].getAngle();
		float2 position = agents[i].getPosition();

		verticle[i * 3].x = position.x + 10 * cos(angle);
		verticle[i * 3].y = position.y + 10 * sin(angle);

		verticle[i * 3 + 1].x = position.x + 6 * cos(angle - radians(90));
		verticle[i * 3 + 1].y = position.y + 6 * sin(angle - radians(90));

		verticle[i * 3 + 2].x = position.x + 6 * cos(angle + radians(90));
		verticle[i * 3 + 2].y = position.y + 6 * sin(angle + radians(90));

		lineVerticle[i * 2].x = position.x;
		lineVerticle[i * 2].y = position.y;

		lineVerticle[i * 2 + 1].x = position.x + 30 * cos(angle);
		lineVerticle[i * 2 + 1].y = position.y + 30 * sin(angle);		
	}
	state[threadIdx.x + blockDim.x * blockIdx.x] = localState;
}


////////////////////////////////////////////////////////////
CudaWandererManager::CudaWandererManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, int blockNumber, float wanderDistance, float wanderRadius, float2 simulationBound)
	: GPUEntityManager(maxSpeed, maxForce, maxAgentNumber, drawMode, threadNumber, blockNumber), m_wanderDistance{ wanderDistance }, m_wanderRadius{ wanderRadius }, m_simulationBound{ simulationBound }
{
}

////////////////////////////////////////////////////////////
void CudaWandererManager::draw(sf::RenderWindow& window)
{
	/// Get projection matrix from sfml camera to adjust to sflm drawing
	glm::mat4 p = glm::make_mat4x4(window.getView().getTransform().getMatrix());

	window.setActive(true);

	m_triangleShader.Activate();

	GLint projectionLoc = glGetUniformLocation(m_triangleShader.ID, "projection");
	glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(p));

	m_agentVAO.Bind();

	glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(3 * m_agents.size()));

	m_agentVAO.Unbind();

	m_lineShader.Activate();

	projectionLoc = glGetUniformLocation(m_lineShader.ID, "projection");
	glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(p));

	m_movementVAO.Bind();

	glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(2 * m_agents.size()));

	m_movementVAO.Unbind();

	window.setActive(false);
}

////////////////////////////////////////////////////////////
void CudaWandererManager::update()
{
	float3* positions;
	float2* linePositions;

	checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaAgentResource, 0));
	checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaMovementResource, 0));
	size_t numBytes{};
	cudaGraphicsResourceGetMappedPointer((void**)&positions, &numBytes, m_cudaAgentResource);
	cudaGraphicsResourceGetMappedPointer((void**)&linePositions, &numBytes, m_cudaMovementResource);

	updateAgent << <m_blockNumber, m_threadNumber >> >
		(positions, linePositions, thrust::raw_pointer_cast(m_agents.data()), static_cast<int>(m_agents.size()),
			m_maxSpeed, m_maxForce, m_wanderDistance, m_wanderDistance, m_state, m_simulationBound);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaAgentResource, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaMovementResource, 0));
}

////////////////////////////////////////////////////////////
bool CudaWandererManager::handleEvent(sf::Event e, sf::RenderWindow& window)
{
	switch (e.type)
	{
		case sf::Event::MouseButtonPressed:
		{
			if (e.key.code == sf::Mouse::Button::Left)
			{
				if (m_agents.size() < m_maxAgentNumber)
				{
					sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition());
					float2 pos{ mousePos.x,mousePos.y };
					m_agents.push_back(Wanderer(pos, RNGGenerator::randFloat(0.0f, 6.2831f)));
				}
				return true;
			}
			break;
		}

		case sf::Event::KeyPressed:
		{
			if (e.key.code == sf::Keyboard::R)
			{
				m_agents.clear();
				sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition());
				std::vector<Wanderer> wandererVec;
				for (int i = 0; i < m_maxAgentNumber; ++i)
				{
					float2 spawnPosition{ RNGGenerator::randFloat(0.0f,m_simulationBound.x),RNGGenerator::randFloat(0.0f,m_simulationBound.y) };
					wandererVec.push_back(Wanderer(spawnPosition, RNGGenerator::randFloat(0.0f, 6.2831f)));
				}
				m_agents.resize(m_maxAgentNumber);
				cudaMemcpy(thrust::raw_pointer_cast(m_agents.data()), wandererVec.data(), sizeof(Wanderer) * m_maxAgentNumber, cudaMemcpyHostToDevice);
			}
		}
	}
	return false;
}