#include "App.h"
#include <DebugPrint.h>

#include <Context.h>
#include <ContextException.h>

#include <GraphicsDevice.h>
#include <GraphicsException.h>
#include <BufferFactory.h>

#include <Input.h>
#include <Util.h>
#include <MeasurementBin.h>

#include <ValueClamp.h>
#include "TempController.h"

#include <iostream> 
#include <amp.h> 
#include <ppl.h>

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h>


#include "PositionSystem.h"
#include "RigidBodySystem.h"
#include "RenderSystem.h"
#include "ControllerSystem.h"
#include "PhysicsWorldHandler.h"

//#define MEASURE_RBODIES

using namespace std;


const double App::DTCAP=0.5;

App::App( HINSTANCE p_hInstance, unsigned int p_width/*=1280*/, unsigned int p_height/*=1024*/ )
{
	int width=p_width,
		height=p_height;
	bool windowMode=true;
	// Context
	try
	{
		m_context = new Context(p_hInstance,"multileg",width,height);
	}
	catch (ContextException& e)
	{
		DEBUGWARNING((e.what()));
	}	
	
	// Graphics
	try
	{
		m_graphicsDevice = new GraphicsDevice(m_context->getWindowHandle(),width,height,windowMode);
	}
	catch (GraphicsException& e)
	{
		DEBUGWARNING((e.what()));
	}


	m_fpsUpdateTick=0.0f;
	m_controller = new TempController(0.0f,10.0f,-50.0f,0.0f);
	m_input = new Input();
	m_input->doStartup(m_context->getWindowHandle());
	m_timeScale = 1.0f;
	m_timeScaleToggle = false;
	m_timePauseStepToggle = false;
	//
	m_startPaused = false;
	//
	m_vp = m_graphicsDevice->getBufferFactoryRef()->createMat4CBuffer();
}

App::~App()
{	
	//SAFE_DELETE(m_kernelDevice);
	SAFE_DELETE(m_graphicsDevice);
	SAFE_DELETE(m_context);
	SAFE_DELETE(m_input);
	SAFE_DELETE(m_controller);
	//
	//delete m_instances;
	delete m_vp;
}

void App::run()
{
	// Set up windows timer
	LARGE_INTEGER countsPerSec = getFrequency();
	double secondsPerCount = 1.0 / (double)countsPerSec.QuadPart;
	// The physics clock is just used to run the physics and runs asynchronously with the gameclock
	LARGE_INTEGER currTimeStamp = getTimeStamp();
	LARGE_INTEGER prevTimeStamp = currTimeStamp;
	// There's an inner loop in here where things happen once every TickMs. These variables are for that.
	LARGE_INTEGER gameClockTimeOffsetStamp = getTimeStamp();
	double gameClockTimeOffset = (double)gameClockTimeOffsetStamp.QuadPart * secondsPerCount;
	const unsigned int gameTickMs = 16;
	double gameTickS = (double)gameTickMs / 1000.0;
	// Absolute start
	double timeStart = (double)getTimeStamp().QuadPart * secondsPerCount;

	// Bullet physics initialization
	// Broadphase object
	btBroadphaseInterface* broadphase = new btDbvtBroadphase();
	// Collision dispatcher with default config
	btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);
	// Register collision algorithm (needed for mesh collisions)
	// btGImpactCollisionAlgorithm::registerAlgorithm(dispatcher);
	// Register physics solver
	// (Single threaded)
	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;
	// ==================================
	// Create the physics world
	// ==================================
	btDiscreteDynamicsWorld* dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
	dynamicsWorld->setGravity(btVector3(0, -9.82f, 0));

	// Measurements
	MeasurementBin<string> rigidBodyStateDbgRecorder;
	// measurer.activate();

	
	// Artemis
	// Create and initialize systems
	artemis::SystemManager * sysManager = m_world.getSystemManager();
	//MovementSystem * movementsys = (MovementSystem*)sm->setSystem(new MovementSystem());
	//addGameLogic(movementsys);
#ifdef MEASURE_RBODIES
	m_rigidBodySystem = (RigidBodySystem*)sysManager->setSystem(new RigidBodySystem(dynamicsWorld, &rigidBodyStateDbgRecorder));
#else
	m_rigidBodySystem = (RigidBodySystem*)sysManager->setSystem(new RigidBodySystem(dynamicsWorld));
#endif
	m_renderSystem = (RenderSystem*)sysManager->setSystem(new RenderSystem(m_graphicsDevice));
	m_controllerSystem = (ControllerSystem*)sysManager->setSystem(new ControllerSystem());
	sysManager->initializeAll();


	

	// Combine Physics with our stuff!
	PhysicsWorldHandler physicsWorldHandler(dynamicsWorld,m_controllerSystem);




	// Entity manager fetch
	artemis::EntityManager * entityManager = m_world.getEntityManager();

	// Create a ground entity

	artemis::Entity & ground = entityManager->create();
	ground.addComponent(new RigidBodyComponent(new btBoxShape(btVector3(400.0f, 10.0f, 400.0f)), 0.0f,
		CollisionLayer::COL_GROUND,CollisionLayer::COL_CHARACTER));
	ground.addComponent(new RenderComponent());
	ground.addComponent(new TransformComponent(glm::vec3(0.0f, -20.0f, 0.0f), 
		glm::quat(glm::vec3(0.0f, 0.0f, 0.0f)),
		glm::vec3(800.0f, 20.0f, 800.0f)));
	ground.refresh();


	// Create axes
	artemis::Entity & axisC = entityManager->create();
	axisC.addComponent(new RenderComponent());
	axisC.addComponent(new TransformComponent(glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(2.0f, 2.0f, 2.0f)));
	axisC.refresh();
	//
	artemis::Entity & axisX = entityManager->create();
	axisX.addComponent(new RenderComponent());
	axisX.addComponent(new TransformComponent(glm::vec3(5.0f, 0.0f, 0.0f),
		glm::vec3(10.0f, 1.0f, 1.0f)));
	axisX.refresh();
	//
	artemis::Entity & axisY = entityManager->create();
	axisY.addComponent(new RenderComponent());
	axisY.addComponent(new TransformComponent(glm::vec3(0.0f, 10.0f, 0.0f),
		glm::vec3(1.0f, 20.0f, 1.0f)));
	axisY.refresh();
	//
	artemis::Entity & axisZ = entityManager->create();
	axisZ.addComponent(new RenderComponent());
	axisZ.addComponent(new TransformComponent(glm::vec3(0.0f, 0.0f, 5.0f),
		glm::vec3(1.0f, 2.0f, 10.0f)));
	axisZ.refresh();


		// Test of controller
		float hipCoronalOffset = 2.0f; // coronal distance between hip joints and center
		for (int x = 0; x < 1; x++) // number of characters
		{
			artemis::Entity & legFrame = entityManager->create();
			glm::vec3 pos = glm::vec3(/*x*3*/0.0f, 10.0f, 50.0f);
			//(float(i) - 50, 10.0f+float(i)*4.0f, float(i)*0.2f-50.0f);
			glm::vec3 lfSize = glm::vec3(hipCoronalOffset*2.0f, 4.0f, 2.0f);
			legFrame.addComponent(new RigidBodyComponent(new btBoxShape(btVector3(lfSize.x, lfSize.y, lfSize.z)*0.5f), 2.0f,
				CollisionLayer::COL_CHARACTER, CollisionLayer::COL_GROUND));
			legFrame.addComponent(new RenderComponent());
			legFrame.addComponent(new TransformComponent(pos,
				glm::quat(glm::vec3(0.0f, 0.0f, 0.0f)),
				lfSize));
			legFrame.refresh();
			//
			vector<artemis::Entity*> hipJoints;
			// Number of leg frames per character
			for (int n = 0; n < 2; n++) // number of legs per frame
			{
				artemis::Entity* prev = &legFrame;
				artemis::Entity* upperLegSegment = NULL;
				float currentHipJointCoronalOffset = (float)(n * 2 - 1)*hipCoronalOffset;
				glm::vec3 legpos = pos + glm::vec3(currentHipJointCoronalOffset, 0.0f, 0.0f);
				glm::vec3 boxSize = glm::vec3(1.0f, 4.0f, 1.0f);
				for (int i = 0; i < 3; i++) // number of segments per leg
				{
					artemis::Entity & childJoint = entityManager->create();
					float jointXOffsetFromParent = 0.0f; // for coronal displacement for hip joints
					float jointZOffsetInChild = 0.0f; // for sagittal displacment for feet
					glm::vec3 parentSz = boxSize;
					boxSize = glm::vec3(1.0f, 4.0f, 1.0f); // set new size for current box
					// segment specific constraint params
					glm::vec3 lowerAngleLim = glm::vec3(-HALFPI, 0.0f, 0.0f);
					glm::vec3 upperAngleLim = glm::vec3(HALFPI, 0.0f, 0.0f);
					if (i == 0) // if hip joint
					{
						upperLegSegment = &childJoint;
						jointXOffsetFromParent = currentHipJointCoronalOffset;
					}
					else if (i == 1) // if knee
					{
						lowerAngleLim = glm::vec3(0.0f, 0.0f, 0.0f);
					}
					else if (i == 2) // if foot
					{
						jointZOffsetInChild = 1.0f;
						boxSize = glm::vec3(1.3f, 1.0f, 2.0f);
						lowerAngleLim = glm::vec3(0.0f, 0.0f, 0.0f);
						upperAngleLim = glm::vec3(0.0f, 0.0f, 0.0f);
					}
					legpos -= glm::vec3(glm::vec3(0.0f, parentSz.y*1.1f, jointZOffsetInChild));
					//(float(i) - 50, 10.0f+float(i)*4.0f, float(i)*0.2f-50.0f);
					childJoint.addComponent(new RigidBodyComponent(new btBoxShape(btVector3(boxSize.x, boxSize.y, boxSize.z)*0.5f), 1.0f, // note, h-lengths
						CollisionLayer::COL_CHARACTER, CollisionLayer::COL_GROUND));
					childJoint.addComponent(new RenderComponent());
					childJoint.addComponent(new TransformComponent(legpos,
						/*glm::quat(glm::vec3(0.0f, 0.0f, 0.0f)), */
						boxSize));					// note scale, so full lengths
					ConstraintComponent::ConstraintDesc constraintDesc{ glm::vec3(0.0f, boxSize.y*0.5f, jointZOffsetInChild),	  // child (this)
						glm::vec3(jointXOffsetFromParent, -parentSz.y*0.5f, 0.0f),													  // parent
						{ lowerAngleLim, upperAngleLim },
						false };
					childJoint.addComponent(new ConstraintComponent(prev, constraintDesc));
					childJoint.refresh();
					prev = &childJoint;
				}
				hipJoints.push_back(upperLegSegment);
			}
			// Controller
			artemis::Entity & controller = entityManager->create();
			//(float(i) - 50, 10.0f+float(i)*4.0f, float(i)*0.2f-50.0f);
			controller.addComponent(new ControllerComponent(&legFrame, hipJoints));
			controller.refresh();
		}



		// Message pump struct
		MSG msg = { 0 };

		// secondary run variable
		// lets non-context systems quit the program
		bool run = true;

		// Dry run, so artemis have run before physics first step
		//gameUpdate(0.0f);
		unsigned int oldSteps = physicsWorldHandler.getNumberOfInternalSteps();
		double time = 0.0;

		while (!m_context->closeRequested() && run)
		{
			if (!pumpMessage(msg))
			{

				// Start by rendering
				render();

				time = (double)getTimeStamp().QuadPart*secondsPerCount - timeStart;

#ifdef MEASURE_RBODIES
				if (time > 5.0)
				{
					rigidBodyStateDbgRecorder.activate();
			}
#endif
				
					// Physics handling part of the loop
					// ========================================================
					/* This, like the rendering, ticks every time around.
					Bullet does the interpolation for us. */
					currTimeStamp = getTimeStamp();
					double phys_dt = (double)m_timeScale*(double)(currTimeStamp.QuadPart - prevTimeStamp.QuadPart) * secondsPerCount;


					//if (rb->isInited())
					//	rb->getRigidBody()->applyForce(btVector3(0.0f, 20.0f, 0.0f), btVector3(0.0f, 0.0f, 0.0f));

					// Tick the bullet world. Keep in mind that bullet takes seconds
					dynamicsWorld->stepSimulation((btScalar)1.0f/60.0f/*phys_dt*//*, 10*/);
					// ========================================================

					unsigned int steps = physicsWorldHandler.getNumberOfInternalSteps();

					prevTimeStamp = currTimeStamp;

#ifdef MEASURE_RBODIES
					if (steps >= 600) run = false;
#endif
					if (steps >= 600) run = false;
				// Game Clock part of the loop
				// ========================================================
				double dt = ((double)getTimeStamp().QuadPart*secondsPerCount - gameClockTimeOffset);
				// Game clock based updates
				while (dt >= gameTickS)
				{
					dt -= gameTickS;
					gameClockTimeOffset += gameTickS;
					// Handle all input
					processInput();
					// Update logic
					double interval = gameTickS;

					handleContext(interval, phys_dt, steps - oldSteps);
					gameUpdate(interval);
				}
				// ========================================================
				oldSteps = physicsWorldHandler.getNumberOfInternalSteps();
		}

	}


#ifdef MEASURE_RBODIES
		rigidBodyStateDbgRecorder.saveMeasurement("Time: "+ToString(time));
		rigidBodyStateDbgRecorder.saveMeasurement("Steps: "+ToString(physicsWorldHandler.getNumberOfInternalSteps()));
#ifndef MULTI
#ifdef _DEBUG
		rigidBodyStateDbgRecorder.saveResults("../output/determinismTest_Debug_STCPU");
#else
		rigidBodyStateDbgRecorder.saveResults("../output/determinismTest_Release_STCPU");
#endif
#else
#ifdef _DEBUG
		rigidBodyStateDbgRecorder.saveResults("../output/determinismTest_Debug_MTCPU");
#else
		rigidBodyStateDbgRecorder.saveResults("../output/determinismTest_Release_MTCPU");
#endif
#endif
#endif


	// Clean up
	entityManager->removeAllEntities();
	delete broadphase;
	delete collisionConfiguration;
	delete dispatcher;
	delete solver;
	delete dynamicsWorld;
}


LARGE_INTEGER App::getTimeStamp()
{
	LARGE_INTEGER stamp;
	QueryPerformanceCounter(&stamp);
	return stamp;
}

LARGE_INTEGER App::getFrequency()
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return freq;
}

void App::updateController(float p_dt)
{
	//m_controller->moveThrust(glm::vec3(0.0f, 0.0f, -0.8f));
	//m_controller->moveAngularThrust(glm::vec3(0.0f, 0.0f, 1.0f)*0.07f);
	// get joystate
	//Just dump the current joy state
	JoyStick* joy = nullptr;
	if (m_input->hasJoysticks()) 
		joy = m_input->g_joys[0];
	float thrustPow = 10.0f;
	// Thrust
	if (m_input->g_kb->isKeyDown(KC_LEFT) || m_input->g_kb->isKeyDown(KC_A))
		m_controller->moveThrust(glm::vec3(-1.0f,0.0f,0.0f)*thrustPow);
	if (m_input->g_kb->isKeyDown(KC_RIGHT) || m_input->g_kb->isKeyDown(KC_D))
		m_controller->moveThrust(glm::vec3(1.0f,0.0f,0.0f)*thrustPow);
	if (m_input->g_kb->isKeyDown(KC_UP) || m_input->g_kb->isKeyDown(KC_W))
		m_controller->moveThrust(glm::vec3(0.0f,1.0f,0.0f)*thrustPow);
	if (m_input->g_kb->isKeyDown(KC_DOWN) || m_input->g_kb->isKeyDown(KC_S))
		m_controller->moveThrust(glm::vec3(0.0f,-1.0f,0.0f)*thrustPow);
	if (m_input->g_kb->isKeyDown(KC_SPACE))
		m_controller->moveThrust(glm::vec3(0.0f,0.0f,1.0f)*thrustPow);
	if (m_input->g_kb->isKeyDown(KC_B))
		m_controller->moveThrust(glm::vec3(0.0f,0.0f,-1.0f)*thrustPow);
	// Joy thrust
	if (joy!=nullptr)
	{
		const JoyStickState& js = joy->getJoyStickState();
		m_controller->moveThrust(glm::vec3((float)(invclampcap(js.mAxes[1].abs,-5000,5000))* 0.0001f,
			(float)(invclampcap(js.mAxes[0].abs,-5000,5000))*-0.0001f,
			(float)(js.mAxes[4].abs)*-0.0001f)*thrustPow);
	}
	
	
	// Angular thrust
	if (m_input->g_kb->isKeyDown(KC_Q) || (joy!=nullptr && joy->getJoyStickState().mButtons[4]))
		m_controller->moveAngularThrust(glm::vec3(0.0f,0.0f,-1.0f));
	if (m_input->g_kb->isKeyDown(KC_E) || (joy!=nullptr && joy->getJoyStickState().mButtons[5]))
		m_controller->moveAngularThrust(glm::vec3(0.0f,0.0f,1.0f));
	if (m_input->g_kb->isKeyDown(KC_T))
		m_controller->moveAngularThrust(glm::vec3(0.0f,1.0f,0.0f));
	if (m_input->g_kb->isKeyDown(KC_R))
		m_controller->moveAngularThrust(glm::vec3(0.0f,-1.0f,0.0f));
	if (m_input->g_kb->isKeyDown(KC_U))
		m_controller->moveAngularThrust(glm::vec3(1.0f,0.0f,0.0f));
	if (m_input->g_kb->isKeyDown(KC_J))
		m_controller->moveAngularThrust(glm::vec3(-1.0f,0.0f,0.0f));
	// Joy angular thrust
	if (joy!=nullptr)
	{
		const JoyStickState& js = joy->getJoyStickState();
		m_controller->moveAngularThrust(glm::vec3((float)(invclampcap(js.mAxes[2].abs,-5000,5000))*-0.00001f,
			(float)(invclampcap(js.mAxes[3].abs,-5000,5000))*-0.00001f,
			0.0f));
	}
	
	float mousemovemultiplier=0.002f;
	float mouseX=(float)m_input->g_m->getMouseState().X.rel*mousemovemultiplier;
	float mouseY=(float)m_input->g_m->getMouseState().Y.rel*mousemovemultiplier;
	if ((abs(mouseX)>0.0f || abs(mouseY)>0.0f) && m_input->g_m->getMouseState().buttonDown(MB_Middle))
	{
		m_controller->rotate(glm::vec3(clamp(mouseY,-1.0f,1.0f),clamp(mouseX,-1.0f,1.0f),0.0f));
	}
}

bool App::pumpMessage( MSG& p_msg )
{
	bool res = PeekMessage(&p_msg, NULL, 0, 0, PM_REMOVE)>0?true:false;
	if (res)
	{
		TranslateMessage(&p_msg);
		DispatchMessage(&p_msg);
	}
	return res;
}


void App::processInput()
{
	m_input->run();
}

void App::handleContext(double p_dt, double p_physDt, unsigned int p_physSteps)
{
	// apply resizing on graphics device if it has been triggered by the context
	if (m_context->isSizeDirty())
	{
		pair<int, int> sz = m_context->getSize();
		m_graphicsDevice->updateResolution(sz.first, sz.second);
	}
	// Print fps in window head border
	m_fpsUpdateTick -= (float)p_dt;
	if (m_fpsUpdateTick <= 0.0f)
	{
		float fps = (1.0f / (float)(p_dt*1000.0f))*1000.0f;
		float pfps = 1.0f / (float)p_physDt;
		m_context->updateTitle((" | Game FPS: " + ToString(fps) + " | Phys steps/frame: " + ToString(p_physSteps) + " | Phys FPS: " + ToString(pfps)).c_str());
		m_fpsUpdateTick = 0.3f;
	}
}

void App::gameUpdate( double p_dt )
{
	float dt = (float)p_dt;
	float game_dt = m_timeScale*(float)p_dt;


	// temp controller update code
	updateController(dt);
	m_controller->setFovFromAngle(52.0f, m_graphicsDevice->getAspectRatio());
	m_controller->update(dt);
	// Get camera info to buffer
	std::memcpy(&m_vp->accessBuffer, &m_controller->getViewProjMatrix(), sizeof(float)* 4 * 4);
	m_vp->update();

	if (m_timePauseStepToggle && m_timeScale > 0.0f)
		m_timeScale = 0.0f;
	if (m_input->g_kb->isKeyDown(KC_RETURN) || m_startPaused)
	{
		if (!m_timeScaleToggle)
		{
			if (m_timeScale < 1.0f)
				m_timeScale = 1.0f;
			else
				m_timeScale = 0.0f;
			m_timeScaleToggle = true;
			m_startPaused = false;
		}
	}
	else
	{
		m_timeScaleToggle = false;
	}
	if (m_input->g_kb->isKeyDown(KC_NUMPAD6))
	{
		if (m_timeScale < 1.0f && !m_timePauseStepToggle)
		{
			m_timePauseStepToggle = true;
			m_timeScale = 1.0f;
		}
	}
	else
	{
		m_timePauseStepToggle = false;
	}
	

	// Update entity systems
	m_world.loopStart();
	m_world.setDelta(game_dt);
	// Physics result gathering have to run first
	m_rigidBodySystem->executeDeferredConstraintInits();
	m_rigidBodySystem->process();
	m_rigidBodySystem->lateUpdate();
	m_controllerSystem->buildCheck(); // leaks
	// // Run all other systems, for which order doesn't matter
	processSystemCollection(&m_orderIndependentSystems);
	// // Render system is processed last
	m_renderSystem->process();
}


void App::render()
{
	// Clear render targets
	m_graphicsDevice->clearRenderTargets();
	// Run passes
	m_graphicsDevice->executeRenderPass(GraphicsDevice::P_COMPOSEPASS);
	m_graphicsDevice->executeRenderPass(GraphicsDevice::P_WIREFRAMEPASS, m_vp, m_renderSystem->getInstances());
	// Flip!
	m_graphicsDevice->flipBackBuffer();										
}

// Add a system for game logic processing
void App::addOrderIndependentSystem(artemis::EntityProcessingSystem* p_system)
{
	m_orderIndependentSystems.push_back(p_system);
}


void App::processSystemCollection(vector<artemis::EntityProcessingSystem*>* p_systemCollection)
{
	unsigned int count = (unsigned int)p_systemCollection->size();
	for (unsigned int i = 0; i < count; i++)
	{
		artemis::EntityProcessingSystem* system = (*p_systemCollection)[i];
		system->process();
	}
}
