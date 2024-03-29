#include "CudaArriverManager.cuh"
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
__global__ void updateAgent(float3* verticle, float2* lineVerticle, Arriver* agents, int taskSize, float2 target,float slowingDistance)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < taskSize; i += blockDim.x * gridDim.x)
	{
		agents[i].arrive(target, slowingDistance);
		agents[i].update(agents[i].getMaxSpeed());

		float angle = agents[i].getAngle();
		float2 position = agents[i].getPosition();

		// Body position
		verticle[i * 3].x = position.x + 10 * cos(angle);
		verticle[i * 3].y = position.y + 10 * sin(angle);

		verticle[i * 3 + 1].x = position.x + 6 * cos(angle - radians(90));
		verticle[i * 3 + 1].y = position.y + 6 * sin(angle - radians(90));

		verticle[i * 3 + 2].x = position.x + 6 * cos(angle + radians(90));
		verticle[i * 3 + 2].y = position.y + 6 * sin(angle + radians(90));

		// Movement vector
		lineVerticle[i * 2].x = position.x;
		lineVerticle[i * 2].y = position.y;

		lineVerticle[i * 2 + 1].x = position.x + 30 * cos(angle);
		lineVerticle[i * 2 + 1].y = position.y + 30 * sin(angle);
	}
}

////////////////////////////////////////////////////////////
CudaArriverManager::CudaArriverManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, int blockNumber, float targetX, float targetY, float slowingDistance)
	: GPUEntityManager(maxSpeed,maxForce,maxAgentNumber,drawMode,threadNumber,blockNumber),m_target{targetX,targetY},m_slowingDistance{slowingDistance}
{
}

////////////////////////////////////////////////////////////
void CudaArriverManager::draw(sf::RenderWindow& window)
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
void CudaArriverManager::update()
{
	// Map openGl resources to cuda
	float3* positions;
	float2* linePositions;
	checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaAgentResource, 0));
	checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaMovementResource, 0));
	size_t numBytes{};
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&positions, &numBytes, m_cudaAgentResource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&linePositions, &numBytes, m_cudaMovementResource));

	// Update agent on gpu
	updateAgent << <m_blockNumber, m_threadNumber >> > (positions, linePositions, thrust::raw_pointer_cast(m_agents.data()), static_cast<int>(m_agents.size()), m_target, m_slowingDistance);

	//Unmap openGL resources from cuda
	checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaAgentResource, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaMovementResource, 0));
}

////////////////////////////////////////////////////////////
bool CudaArriverManager::handleEvent(sf::Event e, sf::RenderWindow& window)
{
	m_target = float2{ window.mapPixelToCoords(sf::Mouse::getPosition()).x,window.mapPixelToCoords(sf::Mouse::getPosition()).y };
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
				m_agents.push_back(Arriver(pos, RNGGenerator::randFloat(0.0f, 6.2831f), RNGGenerator::randFloat(0.01f, m_maxSpeed), RNGGenerator::randFloat(0.01f, m_maxForce)));
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
			std::vector<Arriver> seekerVec;
			for (int i = 0; i < m_maxAgentNumber; ++i)
			{
				float2 spawnPosition{ RNGGenerator::randFloat(mousePos.x - 30000.0f,mousePos.x + 30000.0f),RNGGenerator::randFloat(mousePos.y - 30000.0f,mousePos.y + 30000.f) };
				seekerVec.push_back(Arriver(spawnPosition, RNGGenerator::randFloat(0.0f, 6.2831f), RNGGenerator::randFloat(0.1f, m_maxSpeed), RNGGenerator::randFloat(0.1f, m_maxForce)));
			}
			m_agents.resize(m_maxAgentNumber);
			checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(m_agents.data()), seekerVec.data(), sizeof(Arriver) * m_maxAgentNumber, cudaMemcpyHostToDevice));
		}
	}
	}
	return false;
}