#pragma once
#include <thrust/device_vector.h>
#include "GPUEntityManager.cuh"
#include "Seeker.h"

class CudaSeekerManager : public GPUEntityManager
{
public:
	CudaSeekerManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, int blockNumber, float targetX,float targetY);

	void draw(sf::RenderWindow& window) override;

	void update() override;

	bool handleEvent(sf::Event e, sf::RenderWindow& window) override;

private:
	thrust::device_vector<Seeker> m_agents;		//!< Thrust agent vector
	float2 m_target;							//!< Target position
};