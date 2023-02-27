#pragma once
#include "EntityManager.h"
#include "Wanderer.h"

class WandererManager : public CPUEntityManager
{
public:
	WandererManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, float wanderDistance, float wanderRadius, float2 simulationBound);

	void draw(sf::RenderWindow& window) override;

	void update() override;

	bool handleEvent(sf::Event e, sf::RenderWindow& window) override;

private:
	std::vector<Wanderer>	m_agents;
	float					m_wanderDistance;
	float					m_wanderRadius;
	float2					m_simulationBound;
};