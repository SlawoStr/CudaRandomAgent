#pragma once
#include <SFML/Graphics.hpp>

/// <summary>
/// Abstract class for agent managers
/// </summary>
class EntityManager
{
public:
	enum class DrawMode {
		ALL,		// Draw agents with their movement vectors
		BODY,		// Draw only agents
		MOVEMENT	// Draw only movement vectors
	};


	EntityManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode) : m_maxSpeed(maxSpeed), m_maxForce(maxForce), m_maxAgentNumber(maxAgentNumber), m_drawMode(drawMode)
	{}

	virtual ~EntityManager() {};

	virtual void draw(sf::RenderWindow& window) = 0;

	virtual void update() = 0;

	virtual bool handleEvent(sf::Event e, sf::RenderWindow& window) = 0;

protected:
	float		m_maxSpeed;				//!< Maximum speed
	float		m_maxForce;				//!< Maximum turn force
	size_t		m_maxAgentNumber;		//!< Maximum number of agents in simulation ( for vertex allocation)
	DrawMode	m_drawMode;				//!< Which entities should be drawn on screen
};

/// <summary>
/// Abstract class for cpu agent managers
/// </summary>
class CPUEntityManager : public EntityManager
{
public:
	CPUEntityManager(float maxSpeed, float maxForce, size_t maxAgentNumber, DrawMode drawMode, int threadNumber) : EntityManager{ maxSpeed, maxForce, maxAgentNumber, drawMode }
		, m_threadNumber{ threadNumber }, m_agentVertex{ sf::Triangles,m_maxAgentNumber * 3 }, m_movementVertex{ sf::Lines,m_maxAgentNumber * 2 } 
	{
		for (int i = 0; i < m_maxAgentNumber; ++i)
		{
			m_agentVertex[i * 3].color = sf::Color::Green;
			m_agentVertex[i * 3 + 1].color = sf::Color::Green;
			m_agentVertex[i * 3 + 2].color = sf::Color::Green;

			m_movementVertex[i * 2].color = sf::Color::Red;
			m_movementVertex[i * 2 + 1].color = sf::Color::Red;
		}
	}

	void addThread() { m_threadNumber++; }

	void subThread() { if (m_threadNumber > 1) m_threadNumber--; }

protected:
	sf::VertexArray		m_agentVertex;			//!< Vertex array for agent body drawing
	sf::VertexArray		m_movementVertex;		//!< Vertex array for movement vector drawing
	int					m_threadNumber;			//!< Number of threads
};


