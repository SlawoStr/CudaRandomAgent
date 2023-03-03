#include "ArriverManager.h"
#include "Utilities/RNGGenerator.h"

namespace
{
	/// <summary>
	/// Convert degrees to radians
	/// </summary>
	/// <param name="a">Degree number</param>
	/// <returns>Radians</returns>
	float radians(float a)
	{
		return static_cast<float>(0.017453292 * a);
	}
}

////////////////////////////////////////////////////////////
ArriverManager::ArriverManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, sf::Vector2f target, float slowingDistance)
	: CPUEntityManager(maxSpeed, maxForce, maxAgentNumber, drawMode, threadNumber), m_target(target),m_slowingDistance(slowingDistance)
{}

////////////////////////////////////////////////////////////
void ArriverManager::draw(sf::RenderWindow& window)
{
	#pragma omp parallel for num_threads(m_threadNumber)
	for (int i = 0; i < m_agents.size(); ++i)
	{
		float angle = m_agents[i].getAngle();
		float2 position = m_agents[i].getPosition();
		if (m_drawMode == DrawMode::ALL || m_drawMode == DrawMode::BODY)
		{
			m_agentVertex[i * 3].position = sf::Vector2f(position.x + 10 * cos(angle), position.y + 10 * sin(angle));
			m_agentVertex[i * 3 + 1].position = sf::Vector2f(position.x + 6 * cos(angle + radians(90)), position.y + 6 * sin(angle + radians(90)));
			m_agentVertex[i * 3 + 2].position = sf::Vector2f(position.x + 6 * cos(angle - radians(90)), position.y + 6 * sin(angle - radians(90)));
		}
		if (m_drawMode == DrawMode::ALL || m_drawMode == DrawMode::MOVEMENT)
		{
			m_movementVertex[i * 2].position = sf::Vector2f{ position.x,position.y };
			m_movementVertex[i * 2 + 1].position = sf::Vector2f(position.x + 30 * cos(angle), position.y + 30 * sin(angle));
		}
	}
	// Draw agent body
	if (m_drawMode == DrawMode::ALL || m_drawMode == DrawMode::BODY)
	{
		window.draw(m_agentVertex);
	}
	// Draw movement vector
	if (m_drawMode == DrawMode::ALL || m_drawMode == DrawMode::MOVEMENT)
	{
		window.draw(m_movementVertex);
	}
}

////////////////////////////////////////////////////////////
void ArriverManager::update()
{
	#pragma omp parallel for num_threads(m_threadNumber)
	for (int i = 0; i < m_agents.size(); ++i)
	{
		float2 target{ m_target.x,m_target.y };
		m_agents[i].arrive(target,m_slowingDistance);
		m_agents[i].update(m_agents[i].getMaxSpeed());
	}
}

////////////////////////////////////////////////////////////
bool ArriverManager::handleEvent(sf::Event e, sf::RenderWindow& window)
{
	m_target = window.mapPixelToCoords(sf::Mouse::getPosition());
	switch (e.type)
	{
	case sf::Event::MouseButtonPressed:
	{
		// Spawn Agents
		if (e.key.code == sf::Mouse::Button::Left)
		{
			if (m_agents.size() < m_maxAgentNumber)
			{
				sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition());
				float2 pos{ mousePos.x,mousePos.y };
				m_agents.emplace_back(pos, RNGGenerator::randFloat(0.0f, 6.2831f), RNGGenerator::randFloat(0.01f, m_maxSpeed), RNGGenerator::randFloat(0.01f, m_maxForce));
			}
			return true;
		}
		break;
	}
	case sf::Event::KeyPressed:
	{
		// Reset and spawn maximum number of agents in random positions
		if (e.key.code == sf::Keyboard::R)
		{
			m_agents.clear();
			sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition());
			for (int i = 0; i < m_maxAgentNumber; ++i)
			{
				float2 spawnPosition{ RNGGenerator::randFloat(mousePos.x - 10000.0f,mousePos.x + 10000.0f),RNGGenerator::randFloat(mousePos.y - 10000.0f,mousePos.y + 10000.f) };
				m_agents.emplace_back(spawnPosition, RNGGenerator::randFloat(0.0f, 6.2831f), RNGGenerator::randFloat(0.1f, m_maxSpeed), RNGGenerator::randFloat(0.1f, m_maxForce));
			}
		}
		break;
	}
	}
	return false;
}
