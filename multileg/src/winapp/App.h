#pragma once

#include <windows.h>
#include <glm\gtc\type_ptr.hpp>

#include <InstanceData.h>
#include <CBuffers.h>
#include <Buffer.h>
#include <vector>
#include <Artemis.h>
#include <ResourceManager.h>


class Context;
class GraphicsDevice;
class TempController;
class Input;
class RenderSystem;
class RigidBodySystem;
class ControllerSystem;
class Toolbar;
class DebugDrawer;
class DebugDrawBatch;

using namespace std;

// =======================================================================================
//                                      App
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # App
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class App
{
public:
	App(HINSTANCE p_hInstance, unsigned int p_width=1280, unsigned int p_height=1024);
	virtual ~App();

	void run();
protected:
	Context* m_context;
	GraphicsDevice* m_graphicsDevice;
	DebugDrawer* m_debugDrawer;
	DebugDrawBatch* m_debugDrawBatch;

	void processInput();
	void handleContext(double p_dt, double p_physDt, unsigned int p_physSteps);
	void gameUpdate(double p_dt);

	void addOrderIndependentSystem(artemis::EntityProcessingSystem* p_system);

	void render();
private:
	bool pumpMessage(MSG& p_msg);
	void processSystemCollection(vector<artemis::EntityProcessingSystem*>* p_systemCollection);

	static const double DTCAP;
	float m_fpsUpdateTick;

	void updateController(float p_dt);
	TempController*			m_controller;
	Input*					m_input;
	Toolbar*				m_toolBar;

	// Entity system handling
	artemis::World			m_world;
	vector<artemis::EntityProcessingSystem*> m_orderIndependentSystems;
	// Order dependant systems
	RigidBodySystem*		m_rigidBodySystem;
	RenderSystem*			m_renderSystem;
	ControllerSystem*		m_controllerSystem;

	double m_time;
	float m_timeScale;
	bool m_timeScaleToggle;
	bool m_timePauseStepToggle;
	bool m_triggerPause;
	bool m_gravityStat, m_oldGravityStat;
	bool m_restart;

	// Resource managers
	//ResourceManager<btCollisionShape> m_collisionShapes;

	
	//vector<glm::mat4> m_instanceOrigins;
	//Buffer<glm::mat4>* m_instances;
	Buffer<Mat4CBuffer>* m_vp;
};