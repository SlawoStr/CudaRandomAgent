#include "WandererManager.h"
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
WandererManager::WandererManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber, float wanderDistance, float wanderRadius, float2 simulationBound)
	: CPUEntityManager(maxSpeed,maxForce,maxAgentNumber,drawMode, threadNumber),m_wanderDistance(wanderDistance),m_wanderRadius(wanderRadius),m_simulationBound(simulationBound)
{
}

////////////////////////////////////////////////////////////
void WandererManager::draw(sf::RenderWindow& window)
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
	if (m_drawMode == DrawMode::ALL || m_drawMode == DrawMode::BODY)
	{
		window.draw(m_agentVertex);
	}
	if (m_drawMode == DrawMode::ALL || m_drawMode == DrawMode::MOVEMENT)
	{
		window.draw(m_movementVertex);
	}
}

////////////////////////////////////////////////////////////
void WandererManager::update()
{
	#pragma omp parallel for num_threads(m_threadNumber)
	for (int i = 0; i < m_agents.size(); ++i)
	{
		m_agents[i].wander(m_maxSpeed, m_maxForce, m_wanderDistance, m_wanderRadius, RNGGenerator::randFloat(0.0f, 6.2831f), m_simulationBound);
		m_agents[i].update(m_maxSpeed);
	}
}

////////////////////////////////////////////////////////////
bool WandererManager::handleEvent(sf::Event e, sf::RenderWindow& window)
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
					m_agents.emplace_back(pos, RNGGenerator::randFloat(0.0f, 6.2831f));
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
				for (int i = 0; i < m_maxAgentNumber; ++i)
				{
					float2 spawnPosition{ RNGGenerator::randFloat(0.0f,m_simulationBound.x),RNGGenerator::randFloat(0.0f,m_simulationBound.y) };
					m_agents.emplace_back(spawnPosition, RNGGenerator::randFloat(0.0f, 6.2831f));
				}
			}
			if (e.key.code == sf::Keyboard::B)
			{
				if (m_agents.size() != m_maxAgentNumber)
				{
					size_t spawnNumber{ static_cast<size_t>(m_maxAgentNumber / 10) };
					if (m_agents.size() + spawnNumber > m_maxAgentNumber)
					{
						spawnNumber = m_maxAgentNumber - m_agents.size();
					}
					sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition());
					for (int i = 0; i < spawnNumber; ++i)
					{
						float2 spawnPosition{ mousePos.x,mousePos.y };
						m_agents.emplace_back(spawnPosition, RNGGenerator::randFloat(0.0f, 360.0f));
					}
				}
			}
		}
	}
	return false;
}
