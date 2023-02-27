#pragma once
#include "EntityManager.h"
#include "Arriver.h"

class ArriverManager : public CPUEntityManager
{
public:
	ArriverManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, sf::Vector2f target, float slowingDistance);

	void draw(sf::RenderWindow& window) override;

	void update() override;

	bool handleEvent(sf::Event e, sf::RenderWindow& window) override;

private:
	std::vector<Arriver>	m_agents;
	sf::Vector2f			m_target;
	float					m_slowingDistance;
};
