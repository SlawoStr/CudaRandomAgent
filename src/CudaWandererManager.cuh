#pragma once
#include <thrust/device_vector.h>
#include "GPUEntityManager.cuh"
#include "Wanderer.h"
#include "Utilities/CudaRNG.cuh"

class CudaWandererManager : public GPUEntityManager
{
public:
	CudaWandererManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, int blockNumber, float wanderDistance, float wanderRadius,float2 simulationBound);

	void draw(sf::RenderWindow& window) override;

	void update() override;

	bool handleEvent(sf::Event e, sf::RenderWindow& window) override;

private:
	thrust::device_vector<Wanderer> m_agents;
	float							m_wanderDistance;
	float							m_wanderRadius;
	float2							m_simulationBound;
};