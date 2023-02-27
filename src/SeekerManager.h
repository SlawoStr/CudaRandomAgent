#pragma once
#include "EntityManager.h"
#include "Seeker.h"

class SeekerManager : public CPUEntityManager
{
public:
	SeekerManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, sf::Vector2f target);

	void draw(sf::RenderWindow& window) override;

	void update() override;

	bool handleEvent(sf::Event e, sf::RenderWindow& window) override;

private:
	std::vector<Seeker> m_agents;
	sf::Vector2f		m_target;
};