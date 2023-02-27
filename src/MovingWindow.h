#pragma once
#include "Simulation.h"
#include "CameraController.h"
#include "EntityManager.h"

enum class Procesor
{
	CPU,
	GPU
};

enum class AgentType
{
	ARRIVER,
	SEEKER,
	WANDERER
};

/// <summary>
/// Template for window with basic movement (camera movement, zooming)
/// </summary>
class MovingWindow : public Simulation
{
public:
	/// <summary>
	/// Create window of application
	/// </summary>
	/// <param name="windowWidth">Window width</param>
	/// <param name="windowHeight">Windot height</param>
	/// <param name="windowTitle">Window title</param>
	/// <param name="framerate">Window maximum framerate</param>
	MovingWindow(unsigned windowWidth, unsigned windowHeight, std::string windowTitle, unsigned framerate);
	/// <summary>
	/// Run simulation
	/// </summary>
	void run() override;
private:
	/// <summary>
	/// Handle user interactions
	/// </summary>
	void pollEvent() override;
	/// <summary>
	/// Update elements for next frame
	/// </summary>
	void update() override;
	/// <summary>
	/// Draw elements on screen
	/// </summary>
	void draw() override;
	/// <summary>
	/// Create new agent
	/// </summary>
	void createAgent(Procesor processor, AgentType agentType);
private:
	bool							m_pause{ true };	//!< Is simulation updated
	CameraController				m_camera;			//!< Camera handler for movement etc.
	std::unique_ptr<EntityManager>	m_agentManager;		//!< Agent manager
	sf::RectangleShape				m_background;		//!< Background filler
	Procesor						m_processor;		//!< Currently working processor
	AgentType						m_agentType;		//!< Type of agent
};