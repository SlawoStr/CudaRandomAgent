#pragma once
#include <thrust/device_vector.h>
#include "GPUEntityManager.cuh"
#include "Arriver.h"

class CudaArriverManager : public GPUEntityManager
{
public:
	CudaArriverManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, int blockNumber, float targetX, float targetY, float slowingDistance);

	void draw(sf::RenderWindow& window) override;

	void update() override;

	bool handleEvent(sf::Event e, sf::RenderWindow& window) override;

private:
	thrust::device_vector<Arriver>	m_agents;				//!< Thrust agent vector
	float2							m_target;				//!< Target position
	float							m_slowingDistance;		//!< Distance after which agents begin to slow down
};