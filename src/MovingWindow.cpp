#include "MovingWindow.h"
#include "Utilities/ResourceManager.h"
#include "Utilities/Timer.h"
#include "SeekerManager.h"
#include "ArriverManager.h"
#include "WandererManager.h"
#include "CudaSeekerManager.cuh"
#include "CudaArriverManager.cuh"
#include "CudaWandererManager.cuh"

/// <summary>
/// Map string to agent type enum
/// </summary>
/// <param name="agentType">Agent name enum</param>
/// <returns>Agent type enum</returns>
AgentType getAgentType(std::string agentType)
{
	if (agentType == "Seeker")
	{
		return AgentType::SEEKER;
	}
	else if (agentType == "Arriver")
	{
		return AgentType::ARRIVER;
	}
	else if (agentType == "Wanderer")
	{
		return AgentType::WANDERER;
	}
	return AgentType::SEEKER;
}

/// <summary>
/// Map string to enum type of processor
/// </summary>
/// <param name="processorType">Processor type enum</param>
/// <returns>Processor name enum</returns>
Procesor getProcessorType(std::string processorType)
{
	if (processorType == "CPU")
	{
		return Procesor::CPU;
	}
	else if (processorType == "GPU")
	{
		return Procesor::GPU;
	}
	return Procesor::CPU;
}

////////////////////////////////////////////////////////////
MovingWindow::MovingWindow(unsigned windowWidth, unsigned windowHeight, std::string windowTitle, unsigned framerate) : m_camera(m_window)
{
	m_window.create(sf::VideoMode(windowWidth, windowHeight), windowTitle, sf::Style::Resize);
	m_window.setFramerateLimit(framerate);

	//Set Up glad
	gladLoadGL();

	// Read agents settings from file
	ResourcesManager manager("Resources/AgentConfig.lua");
	getAgentType(manager.getString("AgentType"));
	getProcessorType(manager.getString("Processor"));
	createAgent(getProcessorType(manager.getString("Processor")), getAgentType(manager.getString("AgentType")));

	m_background.setPosition(0.0f, 0.0f);
	m_background.setFillColor(sf::Color::White);
}

////////////////////////////////////////////////////////////
void MovingWindow::run()
{
	Timer t;
	while (m_window.isOpen())
	{
		t.start();
		if (!m_pause)
		{
			update();
		}
		m_window.clear(sf::Color(220, 220, 220));
		draw();
		m_window.display();
		pollEvent();
		t.stop();
		double elapsedTime = t.measure();
		std::cout << "Frame time: " << elapsedTime * 1000 << std::endl;
	}
}

////////////////////////////////////////////////////////////
void MovingWindow::pollEvent()
{
	sf::Event e;
	while (m_window.pollEvent(e))
	{
		if (m_agentManager->handleEvent(e, m_window))
		{
			continue;
		}
		if (m_camera.handleEvent(e))
		{
			continue;
		}
		switch (e.type)
		{
			case sf::Event::Closed:
			{
				m_window.close();
				break;
			}
			case sf::Event::KeyPressed:
			{
				switch (e.key.code)
				{
					case sf::Keyboard::Escape:
					{
						m_window.close();
						break;
					}
					case sf::Keyboard::P:
					{
						m_pause = !m_pause;
						break;
					}
					case sf::Keyboard::M:
					{
						if (m_processor == Procesor::CPU)
						{
							createAgent(Procesor::GPU, m_agentType);
						}
						else
						{
							createAgent(Procesor::CPU, m_agentType);
						}
						break;
					}
					case sf::Keyboard::Num1:
					{
						createAgent(m_processor, AgentType::SEEKER);
						break;
					}
					case sf::Keyboard::Num2:
					{
						createAgent(m_processor, AgentType::ARRIVER);
						break;
					}
					case sf::Keyboard::Num3:
					{
						createAgent(m_processor, AgentType::WANDERER);
						break;
					}
				}
				break;
			}
		}
	}
}

////////////////////////////////////////////////////////////
void MovingWindow::update()
{
	m_agentManager->update();
}

////////////////////////////////////////////////////////////
void MovingWindow::draw()
{
	// Push openGL states to avoid states invalidation by sfml
	m_window.pushGLStates();
	m_window.draw(m_background);
	if (m_processor == Procesor::CPU)
	{
		m_agentManager->draw(m_window);
	}
	m_window.popGLStates();
	if (m_processor == Procesor::GPU)
	{
		m_agentManager->draw(m_window);
	}
}

////////////////////////////////////////////////////////////
void MovingWindow::createAgent(Procesor processor, AgentType agentType)
{
	ResourcesManager manager("Resources/AgentConfig.lua");
	m_agentManager.reset();
	m_agentType = agentType;
	m_processor = processor;

	// Get global managers data
	int threadNumber{ manager.getInt("ThreadNumber") };
	float maxSpeed{ manager.getFloat("MaxSpeed") };
	float maxForce{ manager.getFloat("MaxForce") };
	float wanderDistance{ manager.getFloat("WanderDistance") };
	float wanderRadius{ manager.getFloat("WanderRadius") };
	float slowingDistance{ manager.getFloat("SlowingDistance") };
	size_t maxAgentNumber{ static_cast<size_t>(manager.getInt("MaxAgentNumber")) };
	sf::Vector2f simulationBound{ manager.getFloat("BoundX"),manager.getFloat("BoundY") };
	// Create new agent manager
	switch (m_agentType)
	{
		case AgentType::ARRIVER:
			{
				if (m_processor == Procesor::CPU)
				{
					m_agentManager = std::make_unique<ArriverManager>(maxSpeed, maxForce, maxAgentNumber, EntityManager::DrawMode::ALL, threadNumber, m_window.mapPixelToCoords(sf::Mouse::getPosition()), slowingDistance);
				}
				else
				{
					m_agentManager = std::make_unique<CudaArriverManager>(maxSpeed, maxForce, maxAgentNumber, EntityManager::DrawMode::ALL, 128, 15, m_window.mapPixelToCoords(sf::Mouse::getPosition()).x, m_window.mapPixelToCoords(sf::Mouse::getPosition()).y, slowingDistance);
				}
				break;
			}
		case AgentType::SEEKER:
			{
				if (m_processor == Procesor::CPU)
				{
					m_agentManager = std::make_unique<SeekerManager>(maxSpeed, maxForce, maxAgentNumber, EntityManager::DrawMode::ALL, threadNumber, m_window.mapPixelToCoords(sf::Mouse::getPosition()));
				}
				else
				{
					m_agentManager = std::make_unique<CudaSeekerManager>(maxSpeed, maxForce, maxAgentNumber, EntityManager::DrawMode::ALL, 128, 15, m_window.mapPixelToCoords(sf::Mouse::getPosition()).x, m_window.mapPixelToCoords(sf::Mouse::getPosition()).y);
				}
				break;
			}
		case AgentType::WANDERER:
		{
			if (m_processor == Procesor::CPU)
			{
				float2 simBound{ simulationBound.x,simulationBound.y };
				m_agentManager = std::make_unique<WandererManager>(maxSpeed, maxForce, maxAgentNumber, EntityManager::DrawMode::ALL, threadNumber, wanderDistance, wanderRadius, simBound);
			}
			else
			{
				float2 simBound{ simulationBound.x,simulationBound.y };
				m_agentManager = std::make_unique<CudaWandererManager>(maxSpeed, maxForce, maxAgentNumber, EntityManager::DrawMode::ALL, 128, 150, wanderDistance, wanderRadius, simBound);
			}
			break;
		}
		default:
			break;
	}
	m_background.setSize(simulationBound);
}
